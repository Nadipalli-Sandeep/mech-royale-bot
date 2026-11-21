# Mech Royale bot (mechfight/1). Python standard library only.
FROM python:3.13.2-slim
WORKDIR /bot
COPY . .
RUN python3 -m compileall -q /bot
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
