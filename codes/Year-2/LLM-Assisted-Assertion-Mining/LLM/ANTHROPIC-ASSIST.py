import argparse
import os
from dotenv import load_dotenv
import anthropic
import base64
import datetime
import json

# Load environment variables from .env file
load_dotenv()
SECRET_KEY = os.getenv('ANTHROPIC_API_KEY')
client =  anthropic.Anthropic(api_key=SECRET_KEY)

def getResponse(prompt_template, design_spec, language_model, max_tokens, temp) -> str:
    print("Generating Checks from design specification...")
    
    # Read the prompt template and design specification
    with open(prompt_template, "r") as f:
        prompt = f.read()
    with open(design_spec, "r") as f:
        spec = f.read()
    
    # Create the API request
    message = client.messages.create(
        model=language_model,
        temperature=temp,
        max_tokens=max_tokens,
        system=prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Specification : {spec}"
                    }
                ]
            }
        ]
    )
    return message.content[0].text,  f"{prompt}\n\n{spec}"

def extractJSON(checks, output_path, name, language_model, max_tokens, temp = 0.2) -> None:
    print("Extracting Checks to JSON file...")
    
    # Create the API request
    message = client.messages.create(
        model=language_model,
        temperature=temp,
        max_tokens=max_tokens,
        system="For this given Checklists, you have to extract the List of Checks into a JSON format. Keep the list table from step 3 only into one value as it is, DO NOT CHANGE ANYTHING, DO NOT PROVIDE ANYTHING OTHER THAN THE JSON. Strictly Follow this format:\n{\"chklists\": \"all checks\"}. Remember this text must directly be converted into a python json dictionary using json.loads.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Checks : {checks}"
                    }
                ]
            }
        ]
    )
    
    output_path = os.path.join(output_path, "checklists")
    os.makedirs(output_path, exist_ok=True)
    file_name = f"{name}_JSON_CHECKS_ANTHROPIC_{language_model}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    data = json.loads(message.content[0].text)
    with open(os.path.join(output_path, file_name), "w") as f:
        json.dump(data, f, indent=4)
    print(f"JSON file written to {os.path.join(output_path, file_name)}")


def saveLog(output_path, log, name, model) -> None:
    print("Saving the log to the output file...")
    output_path = os.path.join(output_path, "log")
    os.makedirs(output_path, exist_ok=True)
    file_name = f"{name}_CHECKS_ANTHROPIC_{model}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_log.txt"
    with open(os.path.join(output_path, file_name), "w") as f:
        f.write(log)

def main():
    parser = argparse.ArgumentParser(description="Generates Checker Lists for a Design from its Design using Claude")
    parser.add_argument("-nm", "--name", dest="name", required=True, help="Name of the design")
    parser.add_argument("-p", "--prompt", dest="prompt_template", required=True, help="Prompt template for the analysis")
    parser.add_argument("-d", "--design", dest="design_spec", required=True, help="Design specification")
    parser.add_argument("-tk", "--token", dest="max_tokens", default=8192, help="Maximum tokens for the language model")
    parser.add_argument("-tmp", "--temperature", dest="temperature", default=0.5, help="Temperature of the language model")
    parser.add_argument("-lm", "--langmod", dest="language_model", default="claude-3-5-sonnet-latest", choices= ['claude-3-5-sonnet-latest', 'claude-3-opus-latest', 'claude-3-haiku-latest'], help="Langauge Model for assertion Generation")
    parser.add_argument("-o", "--output", dest="output_path", default="./output", help="Path to save the output")
    
    args = parser.parse_args()
    log = ""
    response, m = getResponse(args.prompt_template, args.design_spec, args.language_model, int(args.max_tokens), float(args.temperature))
    log += "-"*80 + "Propmt" + "-"*80 + "\n\n" + str(m) + "\n\n" + "-"*80 + "Response" + "-"*80 + "\n\n" + str(response) + "\n\n"
    saveLog(args.output_path, log, args.name, args.language_model)
    extractJSON(response, args.output_path, args.name, args.language_model, args.max_tokens, temp = 0.2)
if __name__ == "__main__":
    main()
