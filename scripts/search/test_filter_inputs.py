import asyncio
import json
from search_mcp import filter_search_results

async def run_test(input_value, query='test query'):
    print('\n--- Test input:', type(input_value))
    res = await filter_search_results(input_value, query, top_k=2)
    try:
        parsed = json.loads(res)
    except Exception:
        parsed = res
    print('Result:', parsed)

async def main():
    # 1. JSON string input
    lst =  [{'title': 'President Donald J. Trump - The White House', 'href': 'https://www.whitehouse.gov/administration/donald-j-trump/', 'description': 'Donald J. Trump defines the American success story.'}, {'title': 'Donald Trump | Breaking News & Latest Updates | AP News', 'href': 'https://apnews.com/hub/donald-trump', 'description': 'Stay informed and read the latest breaking news and updates on Donald Trump from AP News, the definitive source for independent journalism.'}, {'title': 'Donald Trump - NBC News', 'href': 'https://www.nbcnews.com/politics/donald-trump', 'description': 'Latest news on President Donald Trump, including updates on his executive orders, administrative decisions from his team, news on his court cases and more.'}]
    js = json.dumps(lst, ensure_ascii=False)
    await run_test(js)

    # 2. Already a Python list
    await run_test(lst)

    # 3. Dict envelope
    test = {}
    test['input_text'] = lst
    await run_test(test)

    # 4. List of message-like objects
    #msgs = [{'id':'1','content': json.dumps(lst),'type':'text'}]
    #await run_test(msgs)

    # 5. Bad input
    #await run_test(123)

if __name__ == '__main__':
    asyncio.run(main())
