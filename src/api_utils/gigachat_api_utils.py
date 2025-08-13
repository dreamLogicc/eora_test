import json
import urllib3
import aiohttp
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


async def query(url: str, headers: dict, payload: dict | str) -> dict:
    """Отправляет асинхронный POST-запрос по указанному URL и возвращает ответ в формате JSON.

    Args:
        url (str): URL-адрес, на который отправляется POST-запрос.
        headers (dict): Заголовки HTTP-запроса.
        payload (dict | str): Тело запроса. Может быть словарём (будет автоматически
            сериализовано в JSON) или строкой.

    Returns:
        dict: Ответ сервера, декодированный из JSON.
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=payload, ssl=False) as response:
            resp = await response.json()
            return resp


async def get_token(client_secret: str) -> str | None:
    """Запрашивает OAuth-токен доступа к GigaChat API через авторизацию с client_secret.

     Args:
         client_secret (str): Секретный ключ

     Returns:
         str: Строка с access_token в случае успешной авторизации.
     """
    url = 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth'
    payload = {'scope': 'GIGACHAT_API_PERS'}
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': 'b83117f0-5720-46ae-8ce1-3b61bef06b1d',
        'Authorization': f'Basic {client_secret}'
    }

    response = await query(url, headers, payload)
    return response['access_token']



async def get_answer(text: str, access_token: str) -> str | None:
    """Отправляет запрос к GigaChat API для генерации ответа на пользовательский текст.

    Args:
        text (str): Входной текст (сообщение от пользователя), на который нужно
            сгенерировать ответ.
        access_token (str): Токен доступа.

    Returns:
        str: Сгенерированный моделью текст ответа
    """
    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
    payload = json.dumps({
        "model": "GigaChat-2-Pro",
        "messages": [
            {
                "role": "user",
                "content": text
            }
        ],
        "temperature": 1,
        "top_p": 0.1,
        "n": 1,
        "stream": False,
        "max_tokens": 512,
        "repetition_penalty": 1
    })

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {access_token}'
    }

    response = await query(url, headers, payload)
    return response['choices'][0]['message']['content']


