import csv
import json

import yaml


def is_json_file(file_path):
    try:
        with open(file_path, newline='') as jsonfile:
            json.loads(jsonfile.read())
            return True
    except ValueError:
        return False


def is_csv_file(file_path):
    with open(file_path, newline='') as csvfile:
        try:
            # dialect = csv.Sniffer().sniff(csvfile.read(1024))
            # Perform various checks on the dialect (e.g., lineseparator,
            # delimiter) to make sure it's sane

            # Don't forget to reset the read position back to the start of
            # the file before reading any entries.
            csvfile.seek(0)
            return True
        except csv.Error:
            # File appears not to be in CSV format; move along
            return False


def is_yaml_file(file_path):
    try:
        with open(file_path, newline='') as yamlfile:
            yaml.safe_load(yamlfile)
            return True
    except yaml.YAMLError:
        return False
