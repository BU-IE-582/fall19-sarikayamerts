# This module enables us to automatically download the latest data available
# In order to run this script, you need to:
# 1- Install pydrive package (https://pypi.org/project/PyDrive/)
# 2- Go to this site and click on "Enable the Drive API"
#    https://developers.google.com/drive/api/v3/quickstart/python
#    Download and rename credentials.json as client_secrets.json and make sure
#    that this json file is in your working directory
#
# DISCLAIMER: NEVER EVER COMMIT THAT JSON FILE TO GIT,
#             OTHERWISE EVERYONE CAN ACCESS YOUR WHOLE GOOGLE DRIVE FILES
#             YOU CANNOT ERASE THAT JSON ONCE YOU COMMIT YOUR FILES IN GIT
#             I SUGGEST YOU TO USE .gitignore OR IF YOU ARE NOT FAMILIAR
#             WITH IT, DO NOT USE THIS MODULE
# 3- Make sure IE582Fall2019_data_files is in your Shared with me page

import os
import logging
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Logger settings, just to provide pretty printing format
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='[%(asctime)s] {%(name)s:%(filename)s:%(lineno)s}'
           ' %(levelname)-4s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

# If there is no folder named data, create to upload new data
if not os.path.exists('data'):
    os.mkdir('data')

# Authenticate the client.
gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

flist = drive.ListFile({
    'q': "title = '{}' and trashed = false"
    .format("IE582Fall2019_data_files")}).GetList()

folder_id = flist[0]['id']

file_list = drive.ListFile({
    'q': "'{}' in parents and trashed=false".format(folder_id)}).GetList()

for files in file_list:
    logger.info('\nData name: {}\nCreated date: {}'
                .format(files['title'], files['createdDate']))
    file = drive.CreateFile({'id': files['id']})
    file.GetContentFile('data/'+files['title'], 'application/zip')
