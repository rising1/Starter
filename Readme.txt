This file keeps the notes to get the cloud platform up and running

pip3 install  -r requirements.txt - ********************* not needed
sudo apt-get install google-cloud-sdk-app-engine-python - ******** not needed
gcloud components install app-engine-python -***************** needed
cd c:\users\phfro\PycharmProjects\Starter ************** needed
gcloud projects create rising1-starter --set-as-default
gcloud services enable cloudbuild.googleapis.com
gcloud app create --project=rising1-starter
gcloud app deploy

gcloud app versions stop 20201031t160121

gcloud projects list
gcloud config set project


to tail the logs

gcloud app logs tail