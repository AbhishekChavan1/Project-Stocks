import requests

api="xWu6XxECpCjH0y5kFjQIKcUvTMSEew_K"
ticker='AMZN'
#ticker=input("Enter ticker: ")
limit=4
year='2024'
month='03'
day='15'
api_url=f"https://api.polygon.io/v2/reference/news?limit={limit}&ticker={ticker}&published_utc=2024-03-15&order=desc&limit=2&sort=published_utc&apiKey=xWu6XxECpCjH0y5kFjQIKcUvTMSEew_K"

data=requests.get(api_url).json()

list=[]
for i in range(int(limit)):
    try:
        news=data['results'][i]['description']
        list.append(news)
    except:
        pass

print(list)