import json


def pretty_print_json(json_data):
    return json.dumps(json_data, indent=4)


def get_pretty_printed_json_response_body(response):
    json_response = json.loads(response.body.decode())
    json_response = json.loads(json_response)
    return json.dumps(response.model_dump_json(), indent=4)
