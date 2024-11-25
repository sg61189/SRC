# LLM Assisted Assertion Mining 

### This is the official repository to setup, build and experiment LLM assisted Formal Verification on difrerent OpenTitan IPs. This project is consisting of Three major parts. 
### 1. Having a FV Tool (eg. JasperGold)
### 2. OpenTitan Setup
### 3. LLM Execution

### Having a FV Tool
## It is important and necessary to have a Formal Verification tool already installed, up and can run upon invoking. Opentitan has it's own set of tools to do this entire thing. For our work we have considered JasperGold.

## OpenTitan Setup
## To setup and work with OpenTitan and it's in-house tools, first we would require python 3.8

## RHEL/CentOS

### Using YUM (RHEL/CentOS 7)
```bash
sudo yum install python38

python3.8 -V
```

### Using DNF (RHEL/CentOS 8)
```bash
wget https://www.python.org/ftp/python/3.8.20/Python-3.8.20.tgz

tar xzf Python-3.8.20.tgz

cd Python-3.8.20

./configure --enable-optimizations --with-ensurepip=install

sudo make altinstall

python3.8 -V
```

## Ubuntu

```bash
sudo apt update

sudo add-apt-repository ppa:deadsnakes/ppa

sudo apt install python3.8=3.8.20*

python3.8 -V
```

### Now after we have Python 3.8 installed, we need to fork the OpenTitan Repo and Install its dependecies

## 1. Clone the OpenTitan Repository
```bash
git clone https://github.com/lowRISC/opentitan.git
```

## 2. Create Virtual Environment

```bash
cd opentitan/

python3.8 -m venv titan-env
```
## 3. Install Dependencies
Activate the Virtual Environment
```bash
# On linux (default):
source titan-env/bin/activate

# On linux (csh):
source titan-env/bin/activate.csh
```

Install Dependencies
```bash
pip install -r python-requirements.txt
```
Export the Repository path to the environment
```bash
# Get the Opentitan Repository Path
pwd

# Export Reposiitory path (Default)
echo 'export REPO_TOP=/path/to/your/repo' >> titan-env/bin/activate
source titan-env/bin/activate

# Export Reposiitory path (csh)
echo 'setenv REPO_TOP /path/to/your/repo' >> titan-env/bin/activate.csh
source titan-env/bin/activate.csh
```

### Congratulations, we have the setup ready for OpenTitan. Now we can run a Formal Check on an example IP of Top Earlgrey design. This will take approximately 50 mins and ensure the setup has been done correctly or not.

```bash
$REPO_TOP/util/dvsim/dvsim.py $REPO_TOP/hw/top_earlgrey/formal/top_earlgrey_fpv_prim_cfgs.hjson --select-cfgs prim_packer_fpv
```
After 50 mins, at the end the std should reflect this output:

```bash
### Branch: master

|      name       |  pass_rate  |  formal_cov  |  stimuli_cov  |  checker_cov  |
|:---------------:|:-----------:|:------------:|:-------------:|:-------------:|
| prim_packer_fpv |  100.00 %   |   86.98 %    |    89.40 %    |    85.25 %    |

          [   legend    ]: [Q: queued, D: dispatched, P: passed, F: failed, K: killed, T: total]                                                                             
00:45:30  [    build    ]: [Q: 0, D: 0, P: 1, F: 0, K: 0, T: 1] 100%   
```
## LLM Setup
Now as we have correctly completed the OpenTitan Setup, Let's move forward with the LLM assistance part.

## Setup Instructions

### Prerequisites

- **Recommended minimum`Python` version:** `3.11.9`

### Deactivate the Opentitan environment and change directory to project root
```bash
deactivate
mkdir LLM
cd LLM
```
### Environment Setup

1. **Create a new virtual environment**
    ```
    python3 -m venv LLM-ENV
    ```
2. **Activate the virtual environment**
    ```
    # For Bash
    source LLM-ENV/bin/activate

    # For csh
    source LLM-ENV/bin/activate.csh
    ```
3. **Install the required packages:** 
   ```
   pip install -r requirements.txt
   ```

## Usage

- ANTHROPIC-ASSIST.py

```bash
usage: ANTHROPIC-ASSIST.py [-h] -nm NAME -p PROMPT_TEMPLATE -d DESIGN_SPEC
                           [-tk MAX_TOKENS] [-tmp TEMPERATURE]
                           [-lm {claude-3-5-sonnet-latest,claude-3-opus-latest,claude-3-haiku-latest}]
                           [-o OUTPUT_PATH]

Generates Checker Lists for a Design from its Design Spec

options:
  -h, --help            show this help message and exit
  -nm NAME, --name NAME
                        Name of the design
  -p PROMPT_TEMPLATE, --prompt PROMPT_TEMPLATE
                        Prompt template for the analysis
  -d DESIGN_SPEC, --design DESIGN_SPEC
                        Design specification
  -tk MAX_TOKENS, --token MAX_TOKENS
                        Maximum tokens for the language model
  -tmp TEMPERATURE, --temperature TEMPERATURE
                        Temperature of the language model
  -lm {claude-3-5-sonnet-latest,claude-3-opus-latest,claude-3-haiku-latest}, --langmod {claude-3-5-sonnet-latest,claude-3-opus-latest,claude-3-haiku-latest}
                        Langauge Model for assertion Generation
  -o OUTPUT_PATH, --output OUTPUT_PATH
                        Path to save the output
```

- OPENAI-ASSIST.py

```bash
usage: OPENAI-ASSIST.py [-h] -nm NAME -p PROMPT_TEMPLATE -d DESIGN_SPEC
                        [-tk MAX_TOKENS] [-tmp TEMPERATURE]
                        [-lm {o1-mini,o1-preview,gpt-4o,gpt-4o-latest,gpt-4o-mini,gpt-4-turbo}]
                        [-o OUTPUT_PATH]

Generates Checker Lists for a Design from its Design using ChatGPT

options:
  -h, --help            show this help message and exit
  -nm NAME, --name NAME
                        Name of the design
  -p PROMPT_TEMPLATE, --prompt PROMPT_TEMPLATE
                        Prompt template for the analysis
  -d DESIGN_SPEC, --design DESIGN_SPEC
                        Design specification
  -tk MAX_TOKENS, --token MAX_TOKENS
                        Maximum tokens for the language model
  -tmp TEMPERATURE, --temperature TEMPERATURE
                        Temperature of the language model
  -lm {o1-mini,o1-preview,gpt-4o,gpt-4o-latest,gpt-4o-mini,gpt-4-turbo}, --langmod {o1-mini,o1-preview,gpt-4o,gpt-4o-latest,gpt-4o-mini,gpt-4-turbo}
                        Langauge Model for assertion Generation
  -o OUTPUT_PATH, --output OUTPUT_PATH
                        Path to save the output
```

- c2a.py

```bash
usage: c2a.py [-h] -nm NAME -p PROMPT_TEMPLATE -r RTL [-des SPEC] -chk CHECKS
              [-tk MAX_TOKENS] [-tmp TEMPERATURE]
              [-lm {claude-3-5-sonnet-latest,claude-3-opus-latest,claude-3-haiku-latest}]
              [-o OUTPUT_PATH]

Generates SystemVerilog assertions based on Design Specification, RTL code,
Verification Checklists (Claude Only) and saves them to a JSON format for later uses.

options:
  -h, --help            show this help message and exit
  -nm NAME, --name NAME
                        Name of the design
  -p PROMPT_TEMPLATE, --prompt PROMPT_TEMPLATE
                        Prompt template for the analysis
  -r RTL, --rtl RTL     RTL File
  -des SPEC, --spec SPEC
                        Design Specification
  -chk CHECKS, --checks CHECKS
                        Previously Generated Checks
  -tk MAX_TOKENS, --token MAX_TOKENS
                        Maximum tokens for the language model
  -tmp TEMPERATURE, --temperature TEMPERATURE
                        Temperature of the language model
  -lm {claude-3-5-sonnet-latest,claude-3-opus-latest,claude-3-haiku-latest}, --langmod {claude-3-5-sonnet-latest,claude-3-opus-latest,claude-3-haiku-latest}
                        Langauge Model for assertion Generation
  -o OUTPUT_PATH, --output OUTPUT_PATH
                        Path to save the output
```

- injectCode.py

```bash
usage: injectCode.py [-h] -r RTL -ast ASTS

Reads assertions from a JSON file and Injects them to a RTL. Also, creates one
backup of the original RTL with a specific hash to be found at project root
directory BACKUP.

options:
  -h, --help            show this help message and exit
  -r RTL, --rtl RTL     RTL File
  -ast ASTS, --assertions ASTS
                        JSON file path for generated assertions
```

- restoreBackup.py (Run this file everytime you make an formal check to restore the RTL to it's original Form)

```bash
usage: restoreBackup.py [-h] random_name

Restore a backup file to its original location.

positional arguments:
  random_name  The randomized name of the backup file to restore.

options:
  -h, --help   show this help message and exit
```

Ensure that your `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are stored in a `.env` file located in the `LLM` directory.

## How to generate assertions for Non-Comportable IPs?

### This is a two step process:
- First we will generate checklists by using this command (Make sure you are in correct folder and with proper environment). We will consider `prim_packer` design for reference: 

  - For claude `python3 ANTHROPIC-ASSIST.py -nm prim_packer -p prompts/intelprompt.txt -d example/prim_packer.md` 
  - For openai `python3 OPENAI-ASSIST.py -nm prim_packer -p prompts/intelprompt.txt -d example/prim_packer.md` 

- Next we will generate assertions by using this command `python3 c2a.py -nm prim_packer -p prompts/checks2aseertionsprompt.txt -r example/prim_packer.sv -chk output/checklists/<file>.json` replace <file> with proper filename

## Okay I have the assertions generated, Now what?

Now we will directly push the generated assertions to the RTL file using a script and run the for verification setup.

- Inject the assertions to the RTL file by using : `python3 injectCode.py -r example/prim_packer.sv -ast output/assertions/<assert>.json` replace <assert> with proper filename

- Copy the RTL file from `example` directory and replace it with the file located at `../hw/ip/prim/rtl/`

- Deactivate the environment by typing `deactivate` and `cd ..`
- Activate the opentitan environment by `source titan-env/bin/activate`
- Run this command to formally check: `$REPO_TOP/util/dvsim/dvsim.py $REPO_TOP/hw/top_earlgrey/formal/top_earlgrey_fpv_prim_cfgs.hjson --select-cfgs prim_packer_fpv` this will produce results
- TBD (Logging Mechanism)

## How to generate assertions for [Comportable IPs](https://opentitan.org/book/doc/contributing/hw/comportability/index.html#document-goals)?

### This is a three step process:

- First we will generate checklists by using this command (Make sure you are in correct folder and with proper environment). We will consider OpenTitan GPIO For Reference:

  - For claude `python3 ANTHROPIC-ASSIST.py -nm prim_packer -p prompts/intelprompt.txt -d ../hw/ip/gpio/doc/theory_of_operation.md`
  - For openai `python3 OPENAI-ASSIST.py -nm prim_packer -p prompts/intelprompt.txt -d ../hw/ip/gpio/doc/theory_of_operation.md`

- Next we will generate assertions by using this command `python3 c2a.py -nm prim_packer -p prompts/checks2aseertionsprompt.txt -r ../hw/ip/gpio/fpv/vip/gpio_assert_fpv.sv -chk output/checklists/<file>.json` replace <file> with proper filename

## Okay I have the assertions generated, Now what?

Now we will directly push the generated assertions to the RTL file using a script and run the for verification setup.

- Deactivate the environment by typing `deactivate` and `cd ..`
- Activate the opentitan environment by `source titan-env/bin/activate`
- Important: Open `../hw/top_earlgrey/formal/top_earlgrey_fpv_ip_cfgs.hjson`. Inside the `use_cfgs` key array, add this block as it is.

```hjson
{
  name: gpio_fpv
  dut: gpio_tb
  fusesoc_core: lowrisc:fpv:gpio_fpv
  import_cfgs: ["{proj_root}/hw/formal/tools/dvsim/common_fpv_cfg.hjson"]
  rel_path: "hw/ip/gpio/{sub_flow}/{tool}"
  defines: "FPV_ALERT_NO_SIGINT_ERR"
  cov: true
}
```
- Run this command to generate a FPV Boilerplate `../util/fpvgen.py ../hw/ip/gpio/rtl/gpio.sv`

- Important: Open `../hw/ip/gpio/fpv/gpio_fpv.core`. Replace the `depend` block with this block.
```core
depend:
      - lowrisc:prim:all
      - lowrisc:ip:gpio:0.1
```

- Inject the assertions to the RTL file by using : `python3 injectCode.py -r ../hw/ip/gpio/fpv/vip/gpio_assert_fpv.sv -ast output/assertions/<assert>.json` replace <assert> with proper filename [TBD]

- Run this command to formally check: `$REPO_TOP/util/dvsim/dvsim.py $REPO_TOP/hw/top_earlgrey/formal/top_earlgrey_fpv_ip_cfgs.hjson --select-cfgs gpio_fpv` this will produce results

- TBD (Logging Mechanism)
