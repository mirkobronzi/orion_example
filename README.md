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

## Note on Orion

Interesting things to remember:

* Orion will always minimize the objective function. So, if you are maximizing
(e.g., f1, accuracy, ..) just pass it with a - symbol before, e.g.,

        report_results([dict(
            name='dev_metric',
            type='objective',
            # note the minus
            value=-best_dev_metric)])
            
* if the experiment is already in the database, you will see the Orion shell.
You can either change the exp. name, or use a different db file.
(or, if you know what you are doing, use the shell to merge your experiments) 

## Note on logs

If no `--log` option is passed (see `main.py`), the logging will happen as usual.

If the `--log` option is passed (see `main.py`), then the root logger is set to print to 
that file.

This is convenient when running with orion, cause we can set that file to be
in the folder that orion is created. (so, the log will be places in the exp. folder).

(this is the behaviour in the example `example_orion_on_cluster_with_slurm`)

Note that any other log (e.g., print statement) will still go on screen.
In SLURM, that means they will go in the SLURM main log file.

If you see the examaple `example_orion_on_cluster_with_slurm/to_submit.sh`, these 
log files are redirected to a folder called `other_logs`.
Most probably, these log files will not contain any useful logging that is not
already in the log file in the exp folder.

(because, HOPEFULLY, your code is using logging() to log, and not print() ...)
