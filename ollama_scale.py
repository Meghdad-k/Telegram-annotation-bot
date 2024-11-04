import asyncio
import aiohttp
import time
import json

# Define the endpoint URL for Ollama
OLLAMA_URL = "http://localhost:11434/api/generate"  # Replace with your Ollama instance URL

# Function to simulate a request to Ollama and measure its processing time
async def send_request(session, request_id):
    start_time = time.time()
    request_data = {
        "model": "aya-expanse:8b",
        "prompt": "یک داستان در مورد پسرک کلاه فروش بنویس",
        "stream": True,
        "options": {"temperature": 0.6, "top_p": 0.9}
    }
    
    async with session.post(OLLAMA_URL, json=request_data) as response:
        processing_time = 0
        async for line in response.content:
            # Decode each line as JSON since it's NDJSON format
            response_data = json.loads(line.decode('utf-8'))
            # You can print or process each response chunk here if needed
            #print(f"Response chunk for Request {request_id}: {response_data}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"Request {request_id} processed in {processing_time:.2f} seconds")
        return processing_time

# Function to simulate concurrent requests and measure total time
async def simulate_requests(n):
    async with aiohttp.ClientSession() as session:
        start_time = time.time()
        tasks = [send_request(session, i) for i in range(n)]
        processing_times = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        print(f"All {n} requests completed in {total_time:.2f} seconds")
        return processing_times, total_time

# Entry point to run the asynchronous simulation
def main(n):
    processing_times, total_time = asyncio.run(simulate_requests(n))
    print("Individual request times:", processing_times)
    print("Total time for all requests:", total_time)

# Run the simulation with the desired number of concurrent requests
if __name__ == "__main__":
    n = 2  # Set the number of concurrent requests here
    main(n)
