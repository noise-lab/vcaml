import argparse
import requests


def get_stats(uuid, token):

    url = f'https://api.zoom.us/v2/metrics/meetings/{uuid}/participants/qos'
    querystring = {"page_size":"38","type":"past"}

    headers = {'authorization': f'Bearer {token}'}

    response = requests.request("GET", url, headers=headers, params=querystring)

    response.json()


def build_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'uuid',
        action='store',
        help='Meeting uuid'
    )

    parser.add_argument(
        'token',
        action='store',
        help='api token'
    )

    return parser


def execute():

    parser = build_parser()
    args = parser.parse_args()

    get_stats(args.uuid, args.token)


if __name__ == "__main__":
    execute()
