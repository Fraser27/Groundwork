#!/usr/bin/env bash
#
# load_firm_metadata.sh — create an Iceberg-backed Glue database of law-firm practice data
# that corroborates the demo documents in sample/legal-demo.zip.
#
# Why it exists: Groundwork governs structured *and* unstructured data, and the interesting
# behaviour is at the join. A conflict check that reads only documents is half the product; one
# that can also answer "is this company a client of ours, and what have we billed them" needs
# rows in a warehouse. Until now there was nothing to point Athena at.
#
# The names are not invented. They are the parties the demo PDFs name, the references those PDFs
# carry, and the charge-out rates the engagement letter quotes. That correlation is the point: a
# governed metric over `matters` should reconcile with a fact extracted from a page, and it cannot
# if the two describe different firms.
#
# One absence is deliberate and load-bearing. **Calder Shipping AG is not in `clients`**, because
# the conflict memorandum says the firm has never acted for it. That is the whole conflict
# analysis: Meridian is a client and a shareholder in Calder, Calder itself is not a client. A
# row for Calder here would contradict the document and quietly make the demo incoherent.
#
# Iceberg rather than a crawler over CSV. A crawler infers a schema from files, which is right
# when somebody else owns the data; here the schema is the thing being demonstrated, so it is
# declared. Iceberg also gives real INSERT and DELETE, so the data loads in the same language as
# the DDL and a re-run converges instead of duplicating.
#
# Usage:
#   ./load_firm_metadata.sh                # create the database, tables and rows
#   DRY_RUN=1 ./load_firm_metadata.sh      # print the SQL, touch nothing
#   DROP_FIRST=1 ./load_firm_metadata.sh   # drop and rebuild the tables

set -euo pipefail

# ---- config (override via env) ---------------------------------------------
GLUE_DB="${GLUE_DB:-groundwork_legal}"
REGION="${REGION:-us-east-1}"
ATHENA_WORKGROUP="${ATHENA_WORKGROUP:-primary}"
DRY_RUN="${DRY_RUN:-0}"
DROP_FIRST="${DROP_FIRST:-0}"

# The Athena results bucket doubles as the Iceberg warehouse. Its lifecycle rule is scoped to
# the athena-results/ prefix precisely so this data survives; an unprefixed 14-day expiry would
# delete the files and leave the Glue metadata behind, which reads as corruption rather than as
# a lifecycle rule working correctly.
ATHENA_BUCKET="${ATHENA_BUCKET:-}"
WAREHOUSE_PREFIX="${WAREHOUSE_PREFIX:-warehouse}"

log () { printf '%s\n' "$*" >&2; }
die () { log "error: $*"; exit 1; }

# Resolved from the script's own location so the script runs from any directory, then falls back
# to whatever python is on PATH: this only needs the standard library.
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
[[ -x "$PYTHON" ]] || die "no python found; set PYTHON=/path/to/python"

# ---- resolve the bucket ----------------------------------------------------
if [[ -z "$ATHENA_BUCKET" ]]; then
  ATHENA_BUCKET=$(aws cloudformation describe-stacks --stack-name GroundworkData --region "$REGION" \
    --query 'Stacks[0].Outputs[?OutputKey==`AthenaResultsBucketName`].OutputValue' \
    --output text 2>/dev/null || true)
fi
[[ -z "$ATHENA_BUCKET" || "$ATHENA_BUCKET" == "None" ]] && \
  die "could not find the Athena results bucket. Set ATHENA_BUCKET=<name>."

ATHENA_OUTPUT="s3://${ATHENA_BUCKET}/athena-results/"
WAREHOUSE="s3://${ATHENA_BUCKET}/${WAREHOUSE_PREFIX}"

log "database:  $GLUE_DB"
log "warehouse: $WAREHOUSE"
log ""

# ---- athena helpers --------------------------------------------------------
run_sql () {  # $1 = SQL, $2 = human label
  local sql="$1" label="${2:-query}"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "--- would run: $label"
    printf '%s\n\n' "$sql"
    return 0
  fi

  local qid state reason
  qid=$(aws athena start-query-execution \
    --region "$REGION" \
    --query-string "$sql" \
    --work-group "$ATHENA_WORKGROUP" \
    --result-configuration "OutputLocation=$ATHENA_OUTPUT" \
    --query 'QueryExecutionId' --output text)

  while :; do
    state=$(aws athena get-query-execution --region "$REGION" --query-execution-id "$qid" \
      --query 'QueryExecution.Status.State' --output text)
    case "$state" in
      SUCCEEDED) log "  ok: $label"; return 0 ;;
      FAILED|CANCELLED)
        reason=$(aws athena get-query-execution --region "$REGION" --query-execution-id "$qid" \
          --query 'QueryExecution.Status.StateChangeReason' --output text)
        log "  FAILED: $label"
        log "          $reason"
        return 1 ;;
      *) sleep 2 ;;
    esac
  done
}

# ---- database --------------------------------------------------------------
if [[ "$DRY_RUN" != "1" ]]; then
  if aws glue get-database --region "$REGION" --name "$GLUE_DB" >/dev/null 2>&1; then
    log "database $GLUE_DB already exists"
  else
    aws glue create-database --region "$REGION" \
      --database-input "{\"Name\":\"$GLUE_DB\",\"Description\":\"Groundwork demo: Thorne Vaux LLP practice data, correlated with the sample documents\"}"
    log "created database $GLUE_DB"
  fi
fi

if [[ "$DROP_FIRST" == "1" ]]; then
  for t in time_entries matters clients; do
    run_sql "DROP TABLE IF EXISTS ${GLUE_DB}.${t}" "drop $t" || true
  done
fi

# ---- tables ----------------------------------------------------------------
#
# Three tables, deliberately few. `clients` is who the firm acts for -- which is the table a
# conflict check actually needs, and the one whose *absences* matter. `matters` is the work.
# `time_entries` is what was billed, which is the number a partner asks about.

run_sql "
CREATE TABLE IF NOT EXISTS ${GLUE_DB}.clients (
  client_id      string,
  client_name    string,
  jurisdiction   string,
  sector         string,
  onboarded_date date,
  risk_rating    string
)
LOCATION '${WAREHOUSE}/clients/'
TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
" "create clients"

run_sql "
CREATE TABLE IF NOT EXISTS ${GLUE_DB}.matters (
  matter_id     string,
  matter_name   string,
  client_id     string,
  adverse_party string,
  practice_area string,
  opened_date   date,
  status        string,
  lead_partner  string
)
LOCATION '${WAREHOUSE}/matters/'
TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
" "create matters"

run_sql "
CREATE TABLE IF NOT EXISTS ${GLUE_DB}.time_entries (
  entry_id    string,
  matter_id   string,
  fee_earner  string,
  grade       string,
  entry_date  date,
  hours       double,
  rate_gbp    double,
  amount_gbp  double,
  narrative   string
)
LOCATION '${WAREHOUSE}/time_entries/'
TBLPROPERTIES ('table_type' = 'ICEBERG', 'format' = 'parquet')
" "create time_entries"

# ---- rows ------------------------------------------------------------------
# Cleared before inserting, so a re-run converges rather than doubling every figure.

run_sql "DELETE FROM ${GLUE_DB}.time_entries" "clear time_entries" || true
run_sql "DELETE FROM ${GLUE_DB}.matters" "clear matters" || true
run_sql "DELETE FROM ${GLUE_DB}.clients" "clear clients" || true

# Ten clients. Northwind, Meridian and Halveston are named in the demo documents as clients;
# Kestrel Bank arranged the Meridian facility and is a client on separate work. The remaining six
# give the data enough spread that a metric grouped by sector or jurisdiction returns more than
# one row. Calder Shipping AG is absent on purpose -- see the header.
run_sql "
INSERT INTO ${GLUE_DB}.clients VALUES
  ('CL-001','Northwind Trading Limited','England & Wales','Commodities',DATE '2019-04-02','medium'),
  ('CL-002','Meridian Bulk Carriers SA','Panama','Shipping',DATE '2021-09-15','high'),
  ('CL-003','Halveston Chartering Limited','England & Wales','Shipping',DATE '2020-01-20','medium'),
  ('CL-004','Kestrel Bank AG','Switzerland','Banking',DATE '2018-06-11','high'),
  ('CL-005','Ardenwood Estates LLP','England & Wales','Real estate',DATE '2020-11-03','low'),
  ('CL-006','Pemberton Hale Group','England & Wales','Manufacturing',DATE '2017-02-27','low'),
  ('CL-007','Saltmarsh Renewables Ltd','Scotland','Energy',DATE '2023-05-19','medium'),
  ('CL-008','Vantage Logistics BV','Netherlands','Logistics',DATE '2021-07-30','medium'),
  ('CL-009','Brightlinger Pharma Inc','Delaware, USA','Life sciences',DATE '2022-10-14','high'),
  ('CL-010','Corviston Aggregates Ltd','England & Wales','Construction',DATE '2019-08-21','low')
" "insert clients"

# Matters come from matters.csv rather than being written out here, because the same ten records
# are also seeded into the graph so documents can be filed against them. Two hand-maintained
# copies of one list diverge, and a matter whose reference differs by one character between the
# warehouse and the graph is precisely the silent non-join this dataset exists to demonstrate
# working.
#
# The first three are the demo documents: same references, same clients, same lead partners, same
# opening dates -- MBC opened 11 September 2024 per the facility summary, NTL engaged 12 March
# 2026 per the engagement letter.
MATTERS_CSV="${MATTERS_CSV:-$(dirname "$0")/matters.csv}"
[[ -f "$MATTERS_CSV" ]] || die "no matter records file at $MATTERS_CSV"

matter_values=$(awk -F, 'NR > 1 && NF > 0 {
  if (NF != 8) { printf("line %d has %d fields, expected 8\n", NR, NF) > "/dev/stderr"; exit 1 }
  # Single quotes would break out of the SQL literal. No field legitimately contains one, so
  # this refuses rather than escaping -- a quote here means the file is not what it claims.
  for (i = 1; i <= NF; i++) if (index($i, "\047")) {
    printf("line %d field %d contains a quote\n", NR, i) > "/dev/stderr"; exit 1
  }
  printf("%s(\047%s\047,\047%s\047,\047%s\047,\047%s\047,\047%s\047,DATE \047%s\047,\047%s\047,\047%s\047)",
         sep, $1, $2, $3, $4, $5, $6, $7, $8)
  sep = ",\n  "
}' "$MATTERS_CSV") || die "could not read $MATTERS_CSV"

[[ -n "$matter_values" ]] || die "$MATTERS_CSV has a header but no records"

run_sql "
INSERT INTO ${GLUE_DB}.matters VALUES
  ${matter_values}
" "insert matters"

# Time entries. Rates on the three document-backed matters are the ones the engagement letter
# quotes -- partner 650, senior associate 420, associate 310, paralegal 165 -- so "what have we
# billed on Northwind" reconciles against a figure a reader can find on the page. The letter
# estimates GBP 180,000 to the end of the hearing, and these entries sit well under it, which is
# the honest state of a matter four months in.
run_sql "
INSERT INTO ${GLUE_DB}.time_entries VALUES
  ('TE-0001','NTL-2026-0114','Perrine Duval','partner',DATE '2026-03-16',4.0,650.0,2600.0,'Engagement review and conflict sign-off'),
  ('TE-0002','NTL-2026-0114','Kavi Iyer','senior associate',DATE '2026-03-18',7.5,420.0,3150.0,'Charterparty and clause 11 withdrawal analysis'),
  ('TE-0003','NTL-2026-0114','Kavi Iyer','senior associate',DATE '2026-03-24',6.0,420.0,2520.0,'Notice-period authorities research'),
  ('TE-0004','NTL-2026-0114','Adaeze Mensah','associate',DATE '2026-03-22',9.0,310.0,2790.0,'Disclosure review including engine logs'),
  ('TE-0005','NTL-2026-0114','Kavi Iyer','senior associate',DATE '2026-03-28',5.5,420.0,2310.0,'Advice on prospects'),
  ('TE-0006','NTL-2026-0114','Perrine Duval','partner',DATE '2026-03-30',2.5,650.0,1625.0,'Review of advice on prospects'),
  ('TE-0007','NTL-2026-0114','Adaeze Mensah','associate',DATE '2026-04-08',6.5,310.0,2015.0,'Quantum schedule and mitigation'),
  ('TE-0008','MBC-2024-0431','James Trelawney','partner',DATE '2024-09-16',8.0,650.0,5200.0,'Facility agreement negotiation'),
  ('TE-0009','MBC-2024-0431','James Trelawney','partner',DATE '2024-10-02',5.0,650.0,3250.0,'Security package and vessel mortgages'),
  ('TE-0010','MBC-2024-0431','Adaeze Mensah','associate',DATE '2024-10-14',11.0,310.0,3410.0,'Charge over Calder shareholding'),
  ('TE-0011','MBC-2024-0431','James Trelawney','partner',DATE '2026-03-09',3.0,650.0,1950.0,'Information barrier implementation'),
  ('TE-0012','HAL-2025-0092','Sian Aldridge','partner',DATE '2026-02-04',5.0,650.0,3250.0,'Know-how note on withdrawal authorities'),
  ('TE-0013','HAL-2025-0092','Adaeze Mensah','associate',DATE '2026-02-06',6.0,310.0,1860.0,'Survey of post-Marisol authorities'),
  ('TE-0014','KBA-2025-0210','James Trelawney','partner',DATE '2025-02-18',11.0,650.0,7150.0,'Restructure term sheet'),
  ('TE-0015','KBA-2025-0210','Adaeze Mensah','associate',DATE '2025-03-04',8.0,310.0,2480.0,'Intercreditor analysis'),
  ('TE-0016','ARD-2024-0055','Miriam Fenwick','partner',DATE '2024-05-20',12.0,600.0,7200.0,'Portfolio due diligence'),
  ('TE-0017','ARD-2024-0055','Priya Nkemelu','paralegal',DATE '2024-06-11',15.0,165.0,2475.0,'Title review across 14 sites'),
  ('TE-0018','PHG-2023-0311','Perrine Duval','partner',DATE '2023-04-03',9.0,620.0,5580.0,'Particulars of claim'),
  ('TE-0019','PHG-2023-0311','Priya Nkemelu','paralegal',DATE '2023-05-19',7.5,165.0,1237.5,'Disclosure exercise'),
  ('TE-0020','SRL-2026-0007','Miriam Fenwick','partner',DATE '2026-01-15',6.0,600.0,3600.0,'Seabed lease heads of terms'),
  ('TE-0021','VLB-2025-0188','Sian Aldridge','partner',DATE '2025-08-26',4.5,650.0,2925.0,'Carriage terms review'),
  ('TE-0022','VLB-2025-0188','Kavi Iyer','senior associate',DATE '2025-09-09',6.5,420.0,2730.0,'Liability cap analysis'),
  ('TE-0023','BPI-2025-0402','Miriam Fenwick','partner',DATE '2025-04-15',8.5,600.0,5100.0,'Licence negotiation'),
  ('TE-0024','COR-2024-0126','Miriam Fenwick','partner',DATE '2024-02-05',5.0,600.0,3000.0,'Quarry lease renewal')
" "insert time_entries"

# ---- primary key hints -----------------------------------------------------
#
# Glue has no first-class primary key, and the metric compiler's fan-out check is silent without
# one -- a join that multiplies rows then inflates a SUM with no warning. The scanner reads the
# `primary_key` table parameter, so it is set here.
#
# Set through the Glue API rather than ALTER TABLE, because Athena rejects table properties it
# does not recognise on an Iceberg table ("Unsupported table property key: primary_key"). Glue is
# where the parameter has to land anyway, since that is where the scanner reads it from.
set_primary_key () {  # $1 = table, $2 = comma-separated columns
  local table="$1" pk="$2"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "--- would set primary_key=$pk on $table"
    return 0
  fi
  local current
  current=$(aws glue get-table --region "$REGION" --database-name "$GLUE_DB" --name "$table" \
    --query 'Table' --output json) || { log "  could not read $table"; return 1; }

  # Merge into the existing parameters. Replacing them would drop metadata_location, which is how
  # Glue finds an Iceberg table at all -- the table would still be listed and no longer readable.
  printf '%s' "$current" | "$PYTHON" -c "
import json, sys
t = json.load(sys.stdin)
params = dict(t.get('Parameters') or {})
params['primary_key'] = '$pk'
keep = {k: t[k] for k in ('Name','StorageDescriptor','PartitionKeys','TableType','Description') if k in t}
keep['Parameters'] = params
json.dump(keep, sys.stdout)
" > /tmp/groundwork-table-input.json || { log "  could not build table input for $table"; return 1; }

  aws glue update-table --region "$REGION" --database-name "$GLUE_DB" \
    --table-input "file:///tmp/groundwork-table-input.json" \
    && log "  ok: primary_key=$pk on $table"
}

set_primary_key matters matter_id || true
set_primary_key clients client_id || true
set_primary_key time_entries entry_id || true

log ""
log "done. In Groundwork: Admin -> Scan catalog, then choose '$GLUE_DB'."
log ""
log "Worth asking afterwards, because each one crosses the structured/unstructured boundary:"
log "  \"what have we billed on NTL-2026-0114?\"        Athena, over time_entries"
log "  \"who is the adverse party on that matter?\"      the graph, from the engagement letter"
log "  \"is Calder Shipping a client of ours?\"          no rows -- which is what the conflict"
log "                                                  memorandum asserts, now checkable"
