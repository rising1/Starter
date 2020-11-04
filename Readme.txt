This file keeps the notes to get the cloud platform up and running

use pipreqs in miniconda to create a requirements.txt file

pip3 install  -r requirements.txt - ********************* not needed
sudo apt-get install google-cloud-sdk-app-engine-python - ******** not needed
gcloud components install app-engine-python -***************** not needed - takes ages
cd c:\users\phfro\PycharmProjects\Starter ************** needed
gcloud projects create rising1-starter --set-as-default ************* needed

Make sure project in google cloud has a billing account

gcloud services enable cloudbuild.googleapis.com
gcloud app create --project=rising1-starter
gcloud app deploy

gcloud app versions stop 20201031t160121

gcloud projects list
gcloud config set project


to tail the logs

gcloud app logs tail