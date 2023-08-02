from dotenv import load_dotenv
from flask import Flask, request, abort
from linebot import (LineBotApi, WebhookHandler)
from linebot.exceptions import (InvalidSignatureError)
from linebot.models import (MessageEvent, TextMessage, TextSendMessage,
                            ImageSendMessage, AudioMessage)
import os
import uuid
import requests

from src.models import OpenAIModel
from src.memory import Memory
from src.logger import logger
from src.storage import Storage, FileStorage, MongoStorage
from src.utils import get_role_and_content
from src.service.youtube import Youtube, YoutubeTranscriptReader
from src.service.website import Website, WebsiteReader
from src.mongodb import mongodb
###
from datetime import datetime
from pymongo.errors import ConnectionFailure
from pymongo import MongoClient
# from sentence_transformers import SentenceTransformer, util

load_dotenv('.env')

app = Flask(__name__)

# modify by owen 20230802, add mongodb and huggingface access secrets
line_bot_api = LineBotApi(os.getenv('LINE_CHANNEL_ACCESS_TOKEN'))
handler = WebhookHandler(os.getenv('LINE_CHANNEL_SECRET'))
mdb_user = os.getenv('MONGODB_USERNAME')
mdb_pass = os.getenv('MONGODB_PASSWORD')
mdb_host = os.getenv('MONGODB_HOST')
mdb_dbs = os.getenv('MONGODB_DATABASE')
hf_token = os.getenv('HUGGINGFACE_TOKEN')
hf_sbert_model = os.getenv('HUGGINGFACE_SBERT_MODEL')

storage = None
youtube = Youtube(step=4)
website = Website()

memory = Memory(system_message=os.getenv('SYSTEM_MESSAGE'),
                memory_message_count=2)
model_management = {}
api_keys = {}

my_secret = os.environ['OPENAI_MODEL_ENGINE']

# comment by owen 20230802
# # Load SBERT model (you can use any SBERT model of your choice)
# sbert_model = SentenceTransformer(
#   'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

###


# Function to save conversation data to MongoDB
def save_conversation_to_mongodb(user_id, user_message, bot_response,
                                 user_timestamp, response_timestamp,
                                 response_time):
  try:
    client = MongoClient('mongodb+srv://'+mdb_user+':'+mdb_pass+'@'+mdb_host)
    db = client[mdb_dbs]
    collection = db['history']

    # Create a document to store the conversation data
    conversation_data = {
      'user_id': user_id,
      'user_message': user_message,
      'bot_response': bot_response,
      'user_timestamp': user_timestamp,
      'response_timestamp': response_timestamp,
      'response_time': response_time,
    }

    # Insert the document into the collection
    collection.insert_one(conversation_data)
    client.close()
  except ConnectionFailure:
    print(f"Failed to connect to MongoDB. Conversation data not saved.")
  except Exception as e:
    print(f"Error while saving conversation data: {str(e)}")


@app.route("/callback", methods=['POST'])
def callback():
  signature = request.headers['X-Line-Signature']
  body = request.get_data(as_text=True)
  app.logger.info("Request body: " + body)
  try:
    handler.handle(body, signature)
  except InvalidSignatureError:
    print(
      "Invalid signature. Please check your channel access token/channel secret."
    )
    abort(400)
  return 'OK'


@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
  user_id = event.source.user_id
  user_message = event.message.text.strip()
  user_timestamp = datetime.now()
  text = event.message.text.strip()
  logger.info(f'{user_id}: {text}')

  try:
    if text.startswith('/註冊'):
      api_key = text[3:].strip()
      model = OpenAIModel(api_key=api_key)
      is_successful, _, _ = model.check_token_valid()
      if not is_successful:
        raise ValueError('Invalid API token')
      model_management[user_id] = model
      storage.save({user_id: api_key})
      msg = TextSendMessage(text='Token 有效，註冊成功')

    elif text.startswith('/指令說明'):
      msg = TextSendMessage(
        text=
        "指令：\n/註冊 + API Token\n👉 API Token 請先到 https://platform.openai.com/ 註冊登入後取得\n\n/系統訊息 + Prompt\n👉 Prompt 可以命令機器人扮演某個角色，例如：請你扮演擅長做總結的人\n\n/清除\n👉 當前每一次都會紀錄最後兩筆歷史紀錄，這個指令能夠清除歷史訊息\n\n/圖像 + Prompt\n👉 會調用 DALL∙E 2 Model，以文字生成圖像\n\n語音輸入\n👉 會調用 Whisper 模型，先將語音轉換成文字，再調用 ChatGPT 以文字回覆\n\n其他文字輸入\n👉 調用 ChatGPT 以文字回覆"
      )

    elif text.startswith('/系統訊息'):
      prompt = text[5:].strip()
      system_prompt = (
        "you are a customer service of an online gaming platform. some players might be upset because they lost some of their money on the platform. However, there is nothing wrong with the platform. It is your job to calm down the customers in traditional Chinese. there are some rules you have to stick to in your answer: \n"
        "1: the response has to be in traditional Chinese\n"
        "2: never reveal your true identity. you are 星城's customer service\n"
        "3: never refer to 星城 or the games it offers as Gambling games")
      memory.change_system_message(user_id, f"{system_prompt}\n\n{prompt}")
      msg = TextSendMessage(text='輸入成功')

    elif text.startswith('/清除'):
      memory.remove(user_id)
      msg = TextSendMessage(text='歷史訊息清除成功')

    elif text.startswith('/圖像'):
      prompt = text[3:].strip()
      memory.append(user_id, 'user', prompt)
      is_successful, response, error_message = model_management[
        user_id].image_generations(prompt)
      if not is_successful:
        raise Exception(error_message)
      url = response['data'][0]['url']
      msg = ImageSendMessage(original_content_url=url, preview_image_url=url)
      memory.append(user_id, 'assistant', url)

    else:
      user_model = model_management[user_id]
      memory.append(user_id, 'user', text)

      # Find the most relevant FAQ answer based on text similarity
      relevant_answer = get_relevant_answer_from_faq(text)

      if relevant_answer:
        msg = TextSendMessage(text=relevant_answer)
        memory.append(user_id, 'assistant', relevant_answer)
        response = relevant_answer
      else:
        url = website.get_url_from_text(text)
        if url:
          if youtube.retrieve_video_id(text):
            is_successful, chunks, error_message = youtube.get_transcript_chunks(
              youtube.retrieve_video_id(text))
            if not is_successful:
              raise Exception(error_message)
            youtube_transcript_reader = YoutubeTranscriptReader(
              user_model, os.getenv('OPENAI_MODEL_ENGINE'))
            is_successful, response, error_message = youtube_transcript_reader.summarize(
              chunks)
            if not is_successful:
              raise Exception(error_message)
            role, response = get_role_and_content(response)
            msg = TextSendMessage(text=response)
          else:
            chunks = website.get_content_from_url(url)
            if len(chunks) == 0:
              raise Exception('無法撈取此網站文字')
            website_reader = WebsiteReader(user_model,
                                           os.getenv('OPENAI_MODEL_ENGINE'))
            is_successful, response, error_message = website_reader.summarize(
              chunks)
            if not is_successful:
              raise Exception(error_message)
            role, response = get_role_and_content(response)
            msg = TextSendMessage(text=response)
            memory.append(user_id, role, response)
        else:
          is_successful, response, error_message = user_model.chat_completions(
            memory.get(user_id), os.getenv('OPENAI_MODEL_ENGINE'))
          if not is_successful:
            raise Exception(error_message)
          # Get role and content from the response
          role, response = get_role_and_content(response)
          msg = TextSendMessage(text=response)
          memory.append(user_id, role, response)

      # Calculate the response time
      response_timestamp = datetime.now()
      response_time = (response_timestamp - user_timestamp).total_seconds()

      # Save the conversation data to MongoDB
      save_conversation_to_mongodb(user_id, user_message, response,
                                   user_timestamp, response_timestamp,
                                   response_time)

  except ValueError:
    msg = TextSendMessage(text='Token 無效，請重新註冊，格式為 /註冊 sk-xxxxx')
  except KeyError:
    msg = TextSendMessage(text='請先註冊 Token，格式為 /註冊 sk-xxxxx')
  except Exception as e:
    memory.remove(user_id)
    if str(e).startswith('Incorrect API key provided'):
      msg = TextSendMessage(text='OpenAI API Token 有誤，請重新註冊。')
    elif str(e).startswith(
        'That model is currently overloaded with other requests.'):
      msg = TextSendMessage(text='已超過負荷，請稍後再試')
    else:
      msg = TextSendMessage(text=str(e))
  line_bot_api.reply_message(event.reply_token, msg)

# add by owen 20230802, query HF sbert API
def hf_sbert_query(payload):

  API_URL = "https://api-inference.huggingface.co/models/" + hf_sbert_model
  headers = {"Authorization": "Bearer " + hf_token}

  response = requests.post(API_URL, headers=headers, json=payload)
  return response.json()


# connect to mongodb FAQ
###
def get_relevant_answer_from_faq(user_question):
  try:
    client = MongoClient('mongodb+srv://'+mdb_user+':'+mdb_pass+'@'+mdb_host)
    db = client[mdb_dbs]
    collection = db['faq']

    # Get all questions from the MongoDB collection
    all_questions = [
      entry['question'] for entry in collection.find({}, {'question': 1})
    ]
    # logger.info(f'{user_id}: {output}')

    # comment by owen, 20230802
    # # Encode the user question using SBERT
    # user_question_embedding = sbert_model.encode([user_question])[0]

    # # Encode the FAQ questions using SBERT
    # faq_question_embeddings = sbert_model.encode(all_questions)

    # # Calculate the similarity between user question and FAQ questions using cosine similarity
    # similarities = util.pytorch_cos_sim(user_question_embedding,
    #                                     faq_question_embeddings)[0]

    # # Find the index of the most similar question
    # most_similar_index = similarities.argmax()

    # # Check if the similarity is above a threshold (you can adjust this threshold as needed)
    # if similarities[most_similar_index] > 0.8:
    #   # Query the MongoDB collection for the corresponding answer to the most similar question
    #   result = collection.find_one(
    #     {"question": all_questions[most_similar_index]})
    #   print(f"Query Results4: {str(result)}")
    # return result
    

    # add by owen, 20230802, compare the similarity between user-input question and frequent questions, through HuggingFace API
    similarity_list = hf_sbert_query({
      "inputs": {
        "source_sentence": user_question,
        "sentences": all_questions
      },
    })
    # print(f"Query Results2: {str(similarity_list)}")

    if max(similarity_list) > 0.6:
      index_of_largest = max(range(len(similarity_list)), key=lambda i: similarity_list[i])
      # print(f"Query Results3: {str(index_of_largest)}")

      # Query the MongoDB collection for the corresponding answer to the most similar question
      answer = collection.find_one({"question": all_questions[index_of_largest]})
      print(f"Query Results4: {str(answer['answer'])}")

      return answer
    # add by owen, end

    else:
      return None

  except ConnectionFailure:
    print("Failed to connect to MongoDB. Unable to retrieve answer.")
    return None
  except Exception as e:
    print(f"Error while querying MongoDB: {str(e)}")
    return None


@handler.add(MessageEvent, message=AudioMessage)
def handle_audio_message(event):
  user_id = event.source.user_id
  audio_content = line_bot_api.get_message_content(event.message.id)
  input_audio_path = f'{str(uuid.uuid4())}.m4a'
  with open(input_audio_path, 'wb') as fd:
    for chunk in audio_content.iter_content():
      fd.write(chunk)

  try:
    if not model_management.get(user_id):
      raise ValueError('Invalid API token')
    else:
      is_successful, response, error_message = model_management[
        user_id].audio_transcriptions(input_audio_path, 'whisper-1')
      if not is_successful:
        raise Exception(error_message)
      memory.append(user_id, 'user', response['text'])
      is_successful, response, error_message = model_management[
        user_id].chat_completions(memory.get(user_id), 'gpt-4')
      if not is_successful:
        raise Exception(error_message)
      role, response = get_role_and_content(response)
      memory.append(user_id, role, response)

      # Save the conversation data to MongoDB
      save_conversation_to_mongodb(user_id, user_message, response['text'],
                                   user_timestamp, response_timestamp,
                                   response_time)

      msg = TextSendMessage(text=response)

  except ValueError:
    msg = TextSendMessage(text='請先註冊你的 API Token，格式為 /註冊 [API TOKEN]')
  except KeyError:
    msg = TextSendMessage(text='請先註冊 Token，格式為 /註冊 sk-xxxxx')
  except Exception as e:
    memory.remove(user_id)
    if str(e).startswith('Incorrect API key provided'):
      msg = TextSendMessage(text='OpenAI API Token 有誤，請重新註冊。')
    else:
      msg = TextSendMessage(text=str(e))
  os.remove(input_audio_path)
  line_bot_api.reply_message(event.reply_token, msg)


@app.route("/", methods=['GET'])
def home():
  return 'Hello World'


if __name__ == "__main__":
  if os.getenv('USE_MONGO'):
    mongodb.connect_to_database()
    storage = Storage(MongoStorage(mongodb.db))
  else:
    storage = Storage(FileStorage('db.json'))
  try:
    data = storage.load()
    for user_id in data.keys():
      model_management[user_id] = OpenAIModel(api_key=data[user_id])
  except FileNotFoundError:
    pass
  app.run(host='0.0.0.0', port=8080)
