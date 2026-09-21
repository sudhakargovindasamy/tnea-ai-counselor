-- ═══════════════════════════════════════════════════════════
-- 10_migrate_branches_schema.sql
-- Migration: Upgrade branches table to support new college_branches_rows.csv schema
-- ═══════════════════════════════════════════════════════════

-- 1. Add new columns if they do not exist
ALTER TABLE branches ADD COLUMN IF NOT EXISTS department_code TEXT;
ALTER TABLE branches ADD COLUMN IF NOT EXISTS department_name TEXT;
ALTER TABLE branches ADD COLUMN IF NOT EXISTS approval_marker TEXT;

-- 2. Backfill from old columns if upgrading an existing deployment
UPDATE branches 
SET department_code = branch_code 
WHERE department_code IS NULL AND branch_code IS NOT NULL;

UPDATE branches 
SET approval_marker = approval_note 
WHERE approval_marker IS NULL AND approval_note IS NOT NULL;

-- 3. Create indexes on new columns for fast lookup
CREATE INDEX IF NOT EXISTS branches_department_code_idx ON branches (department_code);
CREATE INDEX IF NOT EXISTS branches_department_name_idx ON branches (department_name);

-- 4. Reload PostgREST schema cache so Supabase API picks up new columns immediately
NOTIFY pgrst, 'reload schema';

SELECT 'Migration to new branch schema complete!' AS status;

