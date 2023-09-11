"""Metrics

Fetches zoom meeting metrics for specified user/meeting

"""
import netrc
import base64 as b64
import requests
import time
import json
import logging
import datetime
from tinydb import TinyDB
from pathlib import Path
from zoom_api.ztimes import ZoomTime

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO,
                    filename='api.log')

TOKEN_LIFETIME = 3599
MEETINGS_URL = 'https://api.zoom.us/v2/metrics/meetings/{}'
QOS_URL = 'https://api.zoom.us/v2/metrics/meetings/{}/participants/qos'
PARTICIPANTS_URL = 'https://api.zoom.us/v2/metrics/meetings/{}/participants'
PAST_INSTANCES_URL = 'https://api.zoom.us/v2/past_meetings/{}/instances'


class ZoomMetrics:

    def __init__(self):

        # Check if access token exists or has expired
        db = TinyDB((Path(__file__).parent).joinpath('token.json'))
        if len(db.all()) == 0:
            self.token = self.get_token()
        elif time.time() > db.all()[0]['ts'] + TOKEN_LIFETIME:
            self.token = self.refresh_token(db.all()[0]['refresh_token'])
        else:
            self.token = db.all()[0]['access_token']

    def get_code(self):

        auth_tokens = netrc.netrc().authenticators("zoom_api")
        auth_website = (
            f'Visit the following website and enter the resulting url\n\n'
            f'https://zoom.us/oauth/authorize?response_type=code&client_id='
            f'{auth_tokens[0]}&redirect_uri={auth_tokens[1]}\n\n'
        )

        response = input(auth_website)

        return response.split('code=')[1]

    def get_auth(self):

        auth_tokens = netrc.netrc().authenticators("zoom_api")
        client_id = auth_tokens[0]
        client_secret = auth_tokens[2]
        zoom_auth = b64.b64encode(
            f'{client_id}:{client_secret}'.encode('utf-8')).decode('utf-8')

        return zoom_auth, auth_tokens[1]

    def get_token(self):

        code = self.get_code()
        auth, redirect_uri = self.get_auth()
        params = {
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': redirect_uri,
        }

        headers = {
            'Authorization': f'Basic {auth}',
        }

        r = requests.post(url='https://zoom.us/oauth/token',
                          headers=headers, params=params).json()

        db = TinyDB((Path(__file__).parent).joinpath('token.json'))
        db.insert({'access_token': r['access_token'],
                   'refresh_token': r['refresh_token'],
                   'ts': time.time()})

        return r['access_token']

    def refresh_token(self, refresh_token):

        auth, _ = self.get_auth()

        params = {
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token,
        }

        headers = {
            'Authorization': f'Basic {auth}',
        }

        r = requests.post(url='https://zoom.us/oauth/token',
                          headers=headers, params=params).json()

        db = TinyDB((Path(__file__).parent).joinpath('token.json'))
        db.update({'refresh_token': r['refresh_token']})
        db.update({'access_token': r['access_token']})
        db.update({'ts': time.time()})

        return r['access_token']

    def get_meeting_uuid(self, meeting):

        headers = {'authorization': f'Bearer {self.token}'}
        query = {'type': 'past'}

        # Log API request
        logging.info(f'Retrieved meeting {meeting} details')

        # Make API request
        r = requests.get(MEETINGS_URL.format(meeting),
                         headers=headers, params=query).json()

        return r['uuid'], r['topic']

    def get_past_meeting_instances(self, meeting):

        headers = {'authorization': f'Bearer {self.token}'}

        # Log API request
        logging.info(f'Retrieved list of ended meeting instances at {meeting}')

        # Make API request
        r = requests.get(PAST_INSTANCES_URL.format(meeting),
                         headers=headers).json()

        # Sort retrieved meetings and return 
        for meeting in r['meetings']:
            meeting['start_time'] = ZoomTime(
                meeting['start_time']).utc_to_local()

        meetings = sorted(r['meetings'], key=lambda k: k['start_time'])

        return


    def get_participants(self, uuid):

        headers = {'authorization': f'Bearer {self.token}'}
        query = {'type': 'past', 'page_size': '30'}

        # Log API request
        logging.info(f'Retrieved participants list for meeting uuid:{uuid}')

        # Make API request
        r = requests.get(PARTICIPANTS_URL.format(uuid),
                         headers=headers, params=query).json()

        # Return list of unique participants
        attendees = []
        for user in r['participants']:
            attendees.append(user['id'])

        return set(attendees)

    def get_participant_qos(self, meeting):

        meeting_uuid, topic = self.get_meeting_uuid(meeting)
        participants = self.get_participants(meeting_uuid)
        headers = {'authorization': f'Bearer {self.token}'}
        query = {
            "page_size": "30",
            "type": "past",
        }

        # Log API request
        logging.info(f'Retrieved all participant QoS for uuid:{meeting_uuid}')

        # Make API request
        r = requests.get(QOS_URL.format(meeting_uuid),
                         headers=headers, params=query).json()

        fpath = Path(__file__).parent.parent / f'zoom_qoe_metrics/{topic}'

        with open(fpath, 'w') as outfile:
            json.dump(r, outfile)

        return
