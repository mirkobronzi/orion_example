# Goal

Provide examples to use Orion (and mlflow).

Has examples for both local runs and runs on the cluster.

# Install

    python -m venv ve
    
    pip install -r requirements.txt

We will run orion on a db file. You need to create a config file.
* on Linux, that is in `~/.config/orion.core/orion_config.yaml`
* on Mac, that is in `~/Library/Application\ Support/orion.core/orion_config.yaml`
 
The file should contain the following:

(change the `host` part if you like - that file will contain the database)

    database:
      type: 'pickleddb'
      host: '$HOME/orion_db/db.pkl'


# Run the examples

cd to the folder with the example you are interested in (`ls example*`) and run
the `run.sh` file there.

Once done, you can check the logs or check on mlflow the various results.

For mlflow, run `mlflow ui` in the folder where you run the experiment.
Then use your browser to explore the results.
