import os
import xmltojson
import json

# Configuration
input_dir = './trainingFiles'
output_dir = './trainingFilesConverted'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def convert_directory():
    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".xml"):
            xml_path = os.path.join(input_dir, filename)

            # Change .xml to .json for the output file
            json_filename = filename.rsplit('.', 1)[0] + '.json'
            json_path = os.path.join(output_dir, json_filename)

            try:
                # Read the XML file
                with open(xml_path, 'r', encoding='utf-8') as xml_file:
                    xml_content = xml_file.read()

                # Convert XML string to JSON string
                json_data = xmltojson.parse(xml_content)
                json_obj = json.loads(json_data)

                # Cherry pick fields and concatenate section text
                new_obj = {}
                new_obj["drug"] = json_obj["Label"]["@drug"]
                new_obj["text"] = ""
                for section in json_obj["Label"]["Text"]["Section"]:
                    new_obj["text"] += section["#text"]
                new_obj["interactions"] = json_obj["Label"]["LabelInteractions"]["LabelInteraction"]

                # Save to the new JSON file
                with open(json_path, 'w', encoding='utf-8') as json_file:
                    json.dump(new_obj, json_file)

                print(f"Successfully converted: {filename} -> {json_filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    convert_directory()
