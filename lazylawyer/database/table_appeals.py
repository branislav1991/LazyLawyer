from lazylawyer.database import database as db

def write_appeals(appeals):
    """Stores all appeals reference to the database.
    """
    db.batch_insert_check('appeals', appeals, attrs=['orig_case_id'])
