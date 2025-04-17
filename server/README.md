
## Add new IP

gcloud compute firewall-rules create allow-flask-7777-2 \
  --direction=INGRESS \
  --priority=1000 \
  --network=default \
  --action=ALLOW \
  --rules=tcp:7777 \
  --source-ranges=203.0.113.88/32 \
  --target-tags=flask-server-7777

## add Tag to instance

gcloud compute instances add-tags carnegie-mellon-second   --zone=us-central1-a   --tags=flask-server-7777

## Inspect

You can view current firewall rules with:

gcloud compute firewall-rules list


To inspect one:

gcloud compute firewall-rules describe allow-flask-7777