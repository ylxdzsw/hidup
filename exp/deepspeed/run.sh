cd "${BASH_SOURCE%/*}"

WORLDINFO=$(python -c 'import config;import json;print(json.dumps({f"{i}": [*range(config.cards_per_node)] for i in range(config.world_size//config.cards_per_node)}))')
echo $WORLDINFO

unset CPN

python gen_ds_config.py > ds_config.json

python -u -m deepspeed.launcher.launch \
    --world_info="$(base64 <<< "$WORLDINFO")" \
    --node_rank=$NODERANK \
    --master_addr=$(python -c 'import config;print(config.master_addr)') \
    --master_port=$(python -c 'import config;print(config.master_port)') \
    train.py --deepspeed_config=ds_config.json
