

import psycopg2

class ActorCRUD:
    def __init__(self, dbname, user, password, host='hanslab.org'):
        self.conn_params = {
            'dbname': dbname,
            'port':35432,
            'user': user,
            'password': password,
            'host': host,
        }
        self.conn = None
        self.connect()

    def connect(self):
        try:
            self.conn = psycopg2.connect(**self.conn_params)
            print('데이터베이스에 연결되었음')
        except psycopg2.Error as e:
            print(f'데이터베이스 연결 중 오류 발생: {e}')


    def close(self):
        if self.conn is not None:
            self.conn.close()
            print("데이터베이스 연결이 닫혔음")


    def create_actor(self, first_name,last_name):
        print(first_name,last_name)
        with self.conn.cursor() as cur:
            cur.execute("""
                        INSERT INTO actor (first_name,last_name)
                        VALUES (%s,%s) RETURNING actor_id;
                        """, (first_name,last_name,))
            actor_id = cur.fetchone()[0]
            self.conn.commit()
            print(f"배우 '{first_name}{last_name}'가 actor_id {actor_id}로 추가되었음.")
            return actor_id

    def read_actor(self, actor_id):
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM actor WHERE actor_id = %s;", (actor_id,))
            actor = cur.fetchone()
            if actor:
                print(actor)
                return actor
            else:
                print("배우를 찾을 수 없음")
                return None
    def update_actor(self,actor_id, first_name=None,last_name=None):
        with self.conn.cursor() as cur:
            cur.execute("""
                        update actor
                        set first_name = %s, last_name = %s
                        where actor_id = %s;
            """,(first_name, last_name,actor_id))
            self.conn.commit()
            print(f'배우 {actor_id}의 정보가 업데이트 되었음')
             
    
    
    def delete_actor(self, actor_id):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM actor WHERE actor_id = %s;", (actor_id,))
            self.conn.commit()
            print(f'actor {actor_id}의 정보가 삭제됨')

# 예제 실행
actor_crud = ActorCRUD(dbname="dvd_rental_sh", user="postgres", password='likelion12', host='hanslab.org')
actor_id = actor_crud.create_actor("lee","ski")

actor_info = actor_crud.read_actor(actor_id)

actor_up = actor_crud.update_actor(actor_id,first_name='ski',last_name ="lee")

actor_del = actor_crud.delete_actor(actor_id)

actor_crud.close()