from fastapi import APIRouter
import logging

from checkpointer_memory_jobs import CheckpointerMemoryJobs
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create the router
router = APIRouter(prefix="/checkpointer", tags=["Checkpointer"])

############################
## -- LOADER SCHEDULER -- ##
############################

@router.post("/scheduler-overview")
async def scheduler_overview():
    try:
        overview = str(scheduler.scheduler)
        overview_lines = overview.split("\n")
        overview_dict = {
            "max_exec": overview_lines[0].split(",")[0].split("=")[1].strip(),
            "tzinfo": overview_lines[0].split(",")[1].split("=")[1].strip(),
            "priority_function": overview_lines[0].split(",")[2].split("=")[1].strip(),
            "jobs": []
        }
        for line in overview_lines[3:]:
            if line.strip() and not line.startswith("--------"):
                parts = line.split()
                if len(parts) >= 8:
                    job = {
                        "type": parts[0],
                        "function": parts[1],
                        "due_at": f"{parts[2]} {parts[3]}",
                        "tzinfo": parts[4],
                        "due_in": parts[5]
                    }
                    # Replace function names
                    if job["function"] == "#ld_memories(..)":
                        job["function"] = "run_clear_old_memories"
                    
                    # Add "d" to single digit due_in values
                    if job["due_in"].isdigit():
                        job["due_in"] += "d"
                    
                    overview_dict["jobs"].append(job)
                else:
                    job = {
                        "type": parts[0] if len(parts) > 0 else "",
                        "function": parts[1] if len(parts) > 1 else "",
                        "due_at": f"{parts[2]} {parts[3]}" if len(parts) > 3 else "",
                        "tzinfo": parts[4] if len(parts) > 4 else "",
                        "due_in": parts[5] if len(parts) > 5 else ""
                    }
                    # Replace function names
                    if job["function"] == "#ld_memories(..)":
                        job["function"] = "run_clear_old_memories"
                    
                    # Add "d" to single digit due_in values
                    if job["due_in"].isdigit():
                        job["due_in"] += "d"
                    
                    overview_dict["jobs"].append(job)
        return {"overview": overview_dict}
    except Exception as e:
        logger.error(f"Error generating scheduler overview: {e}")
        return {"error": "Failed to generate scheduler overview"}
    

def start_scheduler():
    scheduler.start()

scheduler = CheckpointerMemoryJobs()
scheduler_thread = threading.Thread(target=start_scheduler)
scheduler_thread.start()
