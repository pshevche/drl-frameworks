
## Installation and Setup
    * install postgres
        - TODO: add instructions
    * all calcite dependencies should be downloaded using pom.xml when building
    * all queries are in the repo
    * download imdb dataset
        - official instructions don't seem to work well, but following the
        steps here works:
        - https://github.com/RyanMarcus/imdb_pg_dataset/blob/master/vagrant/config.sh
    * start postgres server with imdb
        - init postgres, createdb etc.
        - postgres -D $DATA_DIR
        - settings should be updated in pg-schema.json
    * Python: Used for all the learning stuff
        - Dependencies: TODO
        - Starting point: src/main/python/main.py
        - Python <-> java communication happens over ZeroMQ sockets. Their port
        numbers need to match (specified in Main.java / and as an argument to
            the python script)

    * Running:
        ```bash
        mvn package
        mvn -e exec:java -Dexec.mainClass=Main
        ```

