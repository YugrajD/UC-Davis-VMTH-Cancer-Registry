"""One-time PostGIS bootstrap for RDS, run via a CDK Custom Resource.

RDS doesn't enable PostGIS by default, and CloudFormation has no native
resource for running SQL against a database - this Lambda connects over the
VPC and runs CREATE EXTENSION IF NOT EXISTS, which is idempotent, so it's
safe to re-run on every stack deploy/update.
"""

import psycopg2


def handler(event, context):
    request_type = event.get("RequestType")
    if request_type == "Delete":
        return {"PhysicalResourceId": "postgis-bootstrap"}

    props = event["ResourceProperties"]
    conn = psycopg2.connect(
        host=props["Host"],
        port=int(props["Port"]),
        dbname=props["DbName"],
        user=props["User"],
        password=props["Password"],
        connect_timeout=10,
    )
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    finally:
        conn.close()

    return {"PhysicalResourceId": "postgis-bootstrap"}
