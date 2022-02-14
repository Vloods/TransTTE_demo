from pydantic import BaseSettings


class Settings(BaseSettings):
    hostname: str = 'transtte.online'
    port: int = 9998
    project_name: str = 'Whoosh'
    ssl_keyfile: str = '/etc/letsencrypt/live/transtte.online/privkey.pem'
    ssl_certfile: str = '/etc/letsencrypt/live/transtte.online/fullchain.pem'

    class Config:
        case_sensitive = False
        env_file = '.env'
        env_file_encoding = 'utf-8'

