import json
import urllib3
import aiohttp

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


async def query(url: str, headers: dict, payload: dict | str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=payload, ssl=False) as response:
            resp = await response.json()
            return resp


async def get_token(client_secret: str) -> str:
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


async def get_answer(text: str, access_token: str) -> str:
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
