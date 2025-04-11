from scheduled_agents import ScheduledAgents

import threading

def start_scheduler():
    scheduler.start()

scheduler = ScheduledAgents()
scheduler_thread = threading.Thread(target=start_scheduler)
scheduler_thread.start()