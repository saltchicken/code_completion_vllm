from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from vllm import LLM
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

# Set up logging
logging.basicConfig(
    filename="generation.log",
    level=logging.INFO,
    # format="%(asctime)s - %(levelname)s - %(message)s",
    format="%(message)s",
)

app = FastAPI()
llm = LLM(model="Qwen/Qwen2.5-Coder-14B-Instruct-AWQ")
executor = ThreadPoolExecutor()


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    do_sample: bool = False


@app.post("/generate")
async def generate(request: GenerateRequest):
    def run_generation():
        outputs = llm.generate(
            [
                {
                    "prompt": request.prompt,
                    "max_tokens": request.max_new_tokens,
                    "sampling_params": {"do_sample": request.do_sample},
                }
            ]
        )
        for output in outputs:
            result = output.outputs[0].text
            logging.info(f"-----Result-----\n{result}")
            # logging.info(
            #     f"\n-----Prompt-----\n{request.prompt}\n\n-----Result-----\n{result}"
            # )
            yield result

    loop = asyncio.get_event_loop()
    generated_chunks = await loop.run_in_executor(
        executor, lambda: list(run_generation())
    )

    async def stream_response():
        for chunk in generated_chunks:
            yield chunk

    return StreamingResponse(stream_response(), media_type="text/plain")
