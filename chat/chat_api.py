from langgraph_sdk import get_client
import asyncio
client = get_client(url="http://localhost:2024")

# Using the graph deployed with the name "agent"
assistant_id = "chat"

async def main():
    # create a thread
    thread = await client.threads.create()
    thread_id = thread["thread_id"]

    # create a streaming run
    async for chunk in client.runs.stream(
        thread_id,
        assistant_id,
        input={
            "messages": [
                {"role": "user", "content": "你好！请简单介绍一下自己。"}
            ]
        },
        stream_mode="updates"
    ):
        print(chunk.data)
        

asyncio.run(main())