import asyncio
from sqlalchemy import select, update, insert, delete
from app.data.models import User

async def main():
    s = select(User)
    u = update(User).values(name="test")
    i = insert(User).values(name="test")
    d = delete(User)
    
    print("select:", s.is_dml)
    print("update:", u.is_dml)
    print("insert:", i.is_dml)
    print("delete:", d.is_dml)

asyncio.run(main())
