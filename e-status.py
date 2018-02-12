# Import smtplib for the actual sending function
import smtplib
import sys
from sh import tail
import sched, time
# Import the email modules we'll need
from email.mime.text import MIMEText


me="axayakac@gmail.com"
#you="iaf@ccg.unam.mx"
you=sys.argv[0]
password=sys.argv[1]

list_files_to_monitor="files_to_mon.txt"
time_lap=10 # In minutes

# Open a plain text file for reading.  For this example, assume that
# the text file contains only ASCII characters.
# Open the list of included files
time_lap=time_lap*60
S = sched.scheduler(time.time, time.sleep)
def do_something(sc): 

    fp = open(list_files_to_monitor, 'r')
    messages=[]
    files=[]
# Create a text/plain message
    for file in fp:
        messages.append(file)
        files.append(file)
        messages.append("\r\n".join(tail(file.strip())))
        messages.append("---------------------------------------------------\n")
            
    msg = MIMEText("\r\n".join(messages))
    fp.close()

# me == the sender's email address
# you == the recipient's email address
    msg['Subject'] = 'The contents of %s' % files
    msg['From'] = me
    msg['To'] = you

# Send the message via our own SMTP server, but don't include the
# envelope header.
    #s = smtplib.SMTP('localhost')
    s = smtplib.SMTP('smtp.gmail.com:587')
    s.ehlo()
    s.starttls()
    s.login(me, password)
    s.sendmail(me, [you], msg.as_string())
    s.quit()

    S.enter(time_lap, 1, do_something, (sc,))

S.enter(time_lap, 1, do_something, (S,))
S.run()
