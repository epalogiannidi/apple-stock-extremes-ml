cwd=$(pwd)
config_path=$(grep "tracking_uri:" configs/config.yaml | awk -F ': ' '{print $2}' | tr -d '"')
tracking_uri="$cwd/$config_path"

echo Starting mlflow server for directory: $tracking_uri --host 0.0.0.0

mlflow ui --backend-store-uri $tracking_uri