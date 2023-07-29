import pathlib
import pandas as pd
import dropbox
from dropbox.exceptions import AuthError

DROPBOX_ACCESS_TOKEN = "sl.BjFD7qU_eA9eoY4nH7_zKpFU7lqHwUsRCUnuUim325ygZl-hwu3q7PHRESRyiBd_xV4pDRXwnnSn1xqkaUmdlbq3p71h6QrRYORbpxMR0idNqsZCRxuo3IQNK5_wo4vwboh48XDqHoL62d3mhJMqccA"


def dropbox_connect():
    """Create a connection to Dropbox."""

    try:
        dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
    except AuthError as e:
        print('Error connecting to Dropbox with access token: ' + str(e))
    return dbx




def dropbox_download_file(dropbox_file_path, local_file_path):
    """Download a file from Dropbox to the local machine."""

    try:
        dbx = dropbox_connect()

        with open(local_file_path, 'wb') as f:
            metadata, result = dbx.files_download(path=dropbox_file_path)
            f.write(result.content)
    except Exception as e:
        print('Error downloading file from Dropbox: ' + str(e))



