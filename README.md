# Setup Instructions

1. Install tshark and allow non-superusers to run captures:

```
sudo apt-get install -y tshark
sudo chmod +x /usr/bin/dumpcap
```

2. Create a virtualenv from the project root and install the python3 requirements:

```
python3 -m venv venv
source venv/bin/activate
pip3 -r install requirements.txt
```

3. Launch jupyter-lab.