#EMPOWER

This is the server-side code or some stuff of the Empower platform.

## Installation

You can install it by having it.

## Requirements

## Usage

Functionalities can be accessed in two ways: 1. from the command-line, 2. through web-requests.

### CLI
`python3 empower_cli.py --help`

In short: 
* `--database`, `--collection`, and `--labels` are mandatory
* --database expects the name of a mongo db
* --collection is a collection in that db
* --labels expects names of columns to predict on, like "something somethingElse"
* Label-Columns' fields should contain exclusively 0 or 1

So for example, if you want to predict if a people have *dogs*, *cats*, and/or *birds*. You have a mongoDB
called **Citizens12**, which has a collection called **livingSituation**. You would call:

`python3 empower_cli.py -db Citizens12 -coll livingSituation -l "has_cat has_dog has_bird"`

Optionals are `--output` and `--correlator`.

--output generates a confusion matrix in /tmp/, where you can find a visual classification report.

--correlator superimposes the values of a column onto the graphs that are generated at each execution. 
For example, if you wanted not only to find out the pet situation above, but also get an idea for the size of
each citizen's house, you could write `-c house-size-in-sqm`.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)