import argparse
import os
from dotenv import load_dotenv
import anthropic
import datetime
import re
import json

# Load environment variables from .env file
load_dotenv()
SECRET_KEY = os.getenv('ANTHROPIC_API_KEY')
client =  anthropic.Anthropic(api_key=SECRET_KEY)

def getResponse(prompt_template, checks, rtl, language_model, max_tokens, temp) -> str:
    print("Generating Assertions From Checks")
    
    # Read the prompt template and design specification
    with open(prompt_template, "r") as f:
        prompt = f.read()
    with open(checks, "r") as f:
        check = json.load(f)
    with open(rtl, "r") as f:
        rtl = f.read()
    rtl = remove_assertions_section(rtl)
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
                        "text": f"Checks : {check['chklists']}\nRTL: \n{rtl}"
                    }
                ]
            }
        ]
    )
    return message.content[0].text,  f"{prompt}\n---Checks---\n{check}\n---RTL---\n{rtl}"

def extractJSON(resp, output_path, name, language_model, max_tokens, temp = 0.2) -> None:
    print("Extracting Assertions to JSON file...")
    
    # Create the API request
    message = client.messages.create(
        model=language_model,
        temperature=temp,
        max_tokens=max_tokens,
        system="For this given response, you have to extract all the code part into a JSON format. Keep the assertions into one dictionary value as it is, DO NOT CHANGE ANYTHING, DO NOT PROVIDE ANYTHING OTHER THAN THE JSON. Strictly Follow this format:\n{\"code\": \"only code part\"}. Remember this text must directly be converted into a python json dictionary using json.loads.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Response : {resp}"
                    }
                ]
            }
        ]
    )
    
    output_path = os.path.join(output_path, "assertions")
    os.makedirs(output_path, exist_ok=True)
    file_name = f"{name}_JSON_ASSERTs_ANTHROPIC_{language_model}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    data = json.loads(message.content[0].text)
    data["code"] = "  //////////////////////////////////////////////\
  // Assertions, Assumptions, and Coverpoints //\
  //////////////////////////////////////////////\
\n\n// LLM\n\n" + data["code"] + "\n\n// LLM\n\n"

    with open(os.path.join(output_path, file_name), "w") as f:
        json.dump(data, f, indent=4)
    print(f"JSON file written to {os.path.join(output_path, file_name)}")

def saveLog(output_path, log, name, model) -> None:
    print("Saving the log to the output file...")
    output_path = os.path.join(output_path, "log")
    os.makedirs(output_path, exist_ok=True)
    file_name = f"{name}_ASSERTIONS_FROM_CHECKS_ANTHROPIC_{model}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_log.txt"
    with open(os.path.join(output_path, file_name), "w") as f:
        f.write(log)

def remove_assertions_section(text):
    """
    Removes content between the assertions header pattern and 'endmodule' keyword.
    
    Args:
        text (str): Input text containing Verilog/SystemVerilog code
        
    Returns:
        str: Modified text with the specified section removed
    """
    pattern = (
        r'^\s*\/\/+\n'
        r'\s*\/\/\s*Assertions,\s*Assumptions,\s*and\s*Coverpoints\s*\/\/\n'
        r'\s*\/\/+\n'
        r'\s*\/\/\s*Assumption:\s*mask_i\s*should\s*be\s*contiguous\s*ones\s*\n'
    )
    
    # Create the full pattern - using raw string for the entire pattern
    full_pattern = r'(' + pattern + r')(.*?)(\n\s*endmodule)'
    
    # Replace the matched content with just endmodule
    result = re.sub(
        full_pattern,
        r'\3',  # Keep only the endmodule part
        RTL,
        flags=re.MULTILINE | re.DOTALL
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate SystemVerilog assertions based on Design Specification, RTL code, Verification Checklists (Claude Only)")
    parser.add_argument("-nm", "--name", dest="name", required=True, help="Name of the design")
    parser.add_argument("-p", "--prompt", dest="prompt_template", required=True, help="Prompt template for the analysis")
    parser.add_argument("-r", "--rtl", dest="rtl", required=True, help="RTL File")
    # parser.add_argument("-des", "--spec", dest="spec", help="Design Specification")
    parser.add_argument("-chk", "--checks", dest="checks", required=True, help="Previously Generated Checks")
    parser.add_argument("-tk", "--token", dest="max_tokens", default=8192, help="Maximum tokens for the language model")
    parser.add_argument("-tmp", "--temperature", dest="temperature", default=0.5, help="Temperature of the language model")
    parser.add_argument("-lm", "--langmod", dest="language_model", default="claude-3-5-sonnet-latest", choices= ['claude-3-5-sonnet-latest', 'claude-3-opus-latest', 'claude-3-haiku-latest'], help="Langauge Model for assertion Generation")
    parser.add_argument("-o", "--output", dest="output_path", default="./output", help="Path to save the output")
    
    args = parser.parse_args()
    log = ""
    response, m = getResponse(args.prompt_template, args.checks, args.rtl, args.language_model, int(args.max_tokens), float(args.temperature))
    log += "-"*80 + "Propmt" + "-"*80 + "\n\n" + str(m) + "\n\n" + "-"*80 + "Response" + "-"*80 + "\n\n" + str(response) + "\n\n"
    extractJSON(response, args.output_path, args.name, args.language_model, args.max_tokens, temp = 0.2)
    saveLog(args.output_path, log, args.name, args.language_model)
if __name__ == "__main__":
    main()
