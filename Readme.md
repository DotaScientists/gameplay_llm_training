

# Prepare the environment: 
Install gcloud
sudo apt-get update -y && sudo apt-get install apt-transport-https ca-certificates gnupg curl -y && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && sudo apt-get update && sudo apt-get install -y google-cloud-cli

Authenticate gcloud
gcloud auth  application-default login
gcloud auth application-default set-quota-project robust-doodad-416318

Install python
sudo apt update && sudo apt upgrade -y && sudo add-apt-repository ppa:deadsnakes/ppa -y && sudo apt update && sudo apt install -y python3.11 && sudo apt install -y python3.11-dev python3.11-venv

python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && rm requirements.txt && pip install -e .

Install docker
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove $pkg; done && sudo install -m 0755 -d /etc/apt/keyrings && sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc && sudo chmod a+r /etc/apt/keyrings/docker.asc  && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null && sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Start training

python -m gameplay_llm_training
