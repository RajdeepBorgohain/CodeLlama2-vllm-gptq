import os
from vllm import SamplingParams
from vllm import LLM
from huggingface_hub import snapshot_download

    # sampling_params = SamplingParams(
    #     n=1,
    #     temperature=0.5,
    #     top_p=1,
    #     use_beam_search=args.use_beam_search,
    #     ignore_eos=True,
    #     max_tokens=256,
    # )



class InferlessPythonModel:
    def initialize(self):
        snapshot_download(
            "TheBloke/CodeLlama-34B-Python-AWQ",
            local_dir="/model",
            token="<<your_token>>",
        )
        self.llm = LLM(
            model="/model",
            quantization="awq",
            # tensor_parallel_size=args.tensor_parallel_size,
            # max_num_seqs=args.batch_size,
            # max_num_batched_tokens=args.batch_size * args.input_len,
            # max_model_len=256,
            # trust_remote_code=args.trust_remote_code,
            dtype="half",
            )
    
    def infer(self, inputs):
        print("inputs[prompt] -->", inputs["prompt"], flush=True)
        prompts = inputs[prompt]
        print("Prompts -->", prompts, flush=True)
        sampling_params = SamplingParams(
            temperature=0.5,
            top_p=1,
            max_tokens=256,
            presence_penalty=1.15,
        )
        result = self.llm.generate(prompts, sampling_params)
        result_output = [output.outputs[0].text for output in result]

        return {"result": result_output[0]}

    def finalize(self, args):
        pass
