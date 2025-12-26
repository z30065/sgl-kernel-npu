import os
import subprocess
import time
import requests
import threading
import psutil
import socket

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_server,
)


KUBE_CONFIG = os.environ.get('KUBECONFIG')
NAMESPACE = os.environ.get('NAMESPACE')
CONFIGMAP_NAME = os.environ.get('KUBE_CONFIG_MAP')
LOCAL_TIMEOUT = 3600
SERVICE_PORT = "6677"

config.load_kube_config(KUBE_CONFIG)
v1 = client.CoreV1Api()

# query configmap
def query_configmap(name, namespace):
    try:
        configmap = v1.read_namespaced_config_map(name, namespace)
        print(f"query_configmap successfully!")
        return configmap
    except ApiException as e:
        print(f"query_configmap error {e=}")
        return None

def run_command(cmd, shell=True):
    try:
        result = subprocess.run(
            cmd, shell=shell, capture_output=True, text=True, check=False
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"command error: {e}")
        return None

# set environment
def set_environment():
    print(f"set environment start ......")
    node_ip = os.getenv("POD_IP")
    hostname = os.getenv("HOSTNAME")
    pod_index = int(hostname.rsplit("-", 1)[-1])
    os.environ["RANK"] = pod_index

    # monitor configmap to generate dist-init-addr and node-rank
    isReady = False
    while not isReady:
        configmap = query_configmap(CONFIGMAP_NAME, NAMESPACE)
        if configmap.data == None:
            print(f"configmap is None, wait for 15s ......")
            time.sleep(15)
            continue
        print(f"monitor {configmap.data=}")

        master_node_ip = None
        for pod_name in configmap.data:
            if pod_name.endswith("sglang-node-0"):
                master_node_ip = configmap.data[pod_name]
                break
        if master_node_ip == None:
            print(f"Can not find master node in configmap: {configmap.data=}")
            continue

        os.environ["MASTER_ADDR"] = master_node_ip
        isReady = True
