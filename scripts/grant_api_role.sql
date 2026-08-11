-- Read-only grants for the catalogue API.
--
-- The pipeline owns the schema and every migration. The API is a reader, and
-- proving it cannot write is easier than trusting that it will not.
--
-- Run as the pipeline owner after `alembic upgrade head`:
--   psql "$DIRECT_DATABASE_URL" -v api_role=catalogue_api -f scripts/grant_api_role.sql
--
-- Role creation and passwords are deployment concerns and are deliberately not
-- literals in this file.

\set api_role :api_role

GRANT USAGE ON SCHEMA public TO :"api_role";

GRANT SELECT ON ALL TABLES IN SCHEMA public TO :"api_role";

-- Without this, a table added by a later migration is invisible to the API
-- until someone remembers to re-run the grant above. That failure surfaces as
-- a 500 in production long after the migration looked successful.
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO :"api_role";
