test_case=$1
cd $SGLANG_SOURCE_PATH
if [ ! -f "${test_case}" ];then
  echo "The test case file is not exist: $test_case"
  exit 0
fi

# speed up by using infra cache services
CACHING_URL="cache-service.nginx-pypi-cache.svc.cluster.local"
sed -Ei "s@(ports|archive).ubuntu.com@${CACHING_URL}:8081@g" /etc/apt/sources.list
pip config set global.index-url http://${CACHING_URL}/pypi/simple
pip config set global.extra-index-url "https://pypi.tuna.tsinghua.edu.cn/simple"
pip config set global.trusted-host "${CACHING_URL} pypi.tuna.tsinghua.edu.cn"

pip3 install kubernetes
pip3 install xgrammar==0.1.25

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

export WORLD_SIZE=2
export HCCL_BUFFSIZE=3000
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0

source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "Running test case ${test_case}"
python3 -u ${test_case}
echo "Finished test case ${test_case}"
