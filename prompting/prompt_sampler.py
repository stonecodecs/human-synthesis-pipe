import random
import sys

class PromptSampler:
    def __init__(self, prompt_filepath, num_samples=10):
        self.prompts = self._load_prompts(prompt_filepath)
        self.num_samples = num_samples
        if not self.prompts:
            raise ValueError("The file contains no valid prompts.")
    
    def _load_prompts(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        return prompts
    
    def sample_prompt(self, num_samples=None):
        """Randomly sample and return a prompt."""
        return random.sample(self.prompts, num_samples if num_samples is not None else self.num_samples)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python prompt_sampler.py <filepath>")
        sys.exit(1)
    filepath = sys.argv[1]
    prompt_sampler = PromptSampler(filepath, num_samples=10)
    samples = prompt_sampler.sample_prompt()
    for sample in samples:
        print(sample)
