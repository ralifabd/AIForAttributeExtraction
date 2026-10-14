#Gigachat -> Облачный LLM, который будет возращать ответ на вопросы по Системному PROMT
import requests
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

ClientID = '01a139ff-1358-70ee-8f72-81cae139d05c'
AuthorizationKey = 'MDFhMTM5ZmYtMTM1OC03MGVlLThmNzItODFjYWUxMzlkMDVjOmJmYTc3NmM0LWNhYmYtNDU2ZC05MzlkLTkzNmI1NzdlMDU5NQ=='

# Будем использовать SDK
def getAnswer():
    client = GigaChat(
        base_url="https://api.giga.chat/v1",
        credentials=AuthorizationKey,
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=False,
    )

    PROMPT = "Привет! Как дела?"

    chat = Chat(
        model="GigaChat-2-Max",
        messages=[Messages(role=MessagesRole.USER, content=PROMPT)],
    )
    resp = client.chat(chat)
    return resp

print(getAnswer())




# ТЕСТИК
# Получение токена, токен действует 30 минут только и все! Можно нврн в кеше хранить - но хз
'''def getToken():
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    payload={
      'scope': 'GIGACHAT_API_PERS'
    }
    headers = {
      'Content-Type': 'application/x-www-form-urlencoded',
      'Accept': 'application/json',
      'RqUID': ClientID,
      'Authorization': 'Basic ' + AuthorizationKey
    }
    # обработать ошибки
    response = requests.request("POST", url, headers=headers, data=payload, verify=False)
    return response

def getModels():
    #Получение доступных моделей
    url = "https://api.giga.chat/v1/models"
    payload={}
    headers = {
      'Accept': 'application/json',
      'Authorization': 'Bearer ' + access_token
    }

    response = requests.request("GET", url, headers=headers, data=payload, verify=False)
    # обработать ошибки
    return response

response1 = getToken()
access_token = response1.json()['access_token']
'''