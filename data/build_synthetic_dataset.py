"""Generate the built-in synthetic SQL-injection dataset.

Produces `data/synthetic_dataset.csv` with columns: text,label
label: 1 = SQL injection payload, 0 = benign SQL / normal text input.
Run: python data/build_synthetic_dataset.py
"""
import csv
from pathlib import Path

INJECTIONS = [
    # classic login bypass / tautology
    "' OR 1=1--",
    "' OR 1=1 --",
    "'OR 1=1#",
    "' OR '1'='1",
    '" OR ""=""',
    "admin'--",
    "admin' #",
    "admin'/*",
    "or 1=1;--",
    "1 or 1=1",
    "-1 OR 1=1",
    "' or 1=1 limit 1 --",
    "') or ('1'='1",
    # union based
    "' UNION SELECT username, password FROM users--",
    "1 UNION SELECT 1,2,3--",
    "' union select null, null, null--",
    "union all select user, pass from admins",
    "1 UNION ALL SELECT table_name FROM information_schema.tables--",
    "' UNION SELECT column_name FROM information_schema.columns WHERE table_name='users'--",
    "-1 UNION SELECT 1,@@version,3--",
    # case / obfuscation
    "UnIoN sElEcT 1,2,3--",
    "oR 1=1",
    "AdMiN'--",
    "SELECT/**/1",
    "'/**/OR/**/1=1--",
    "1/**/AND/**/1=1",
    "%27%20OR%201=1--",
    "%27%09OR%09%271%27=%271",
    "'%20or%201=1%20--",
    # char() / hex
    "0x61646D696E",
    "admin' AND 1=1-- 0x27",
    "char(39,111,114,39)",
    "' OR username=char(97,100,109,105,110)--",
    # comment / trailing space tricks
    "admin'-- -",
    "' OR '1'='1'/*",
    "1' and 1=1 /* inline */ --",
    # boolean blind
    "1' AND '1'='1",
    "1' AND '1'='2",
    "1 AND 1=1",
    "1' AND 1=1--",
    "1' AND 1=2--",
    "user' AND SUBSTRING(@@version,1,1)='5",
    "1' AND ASCII(SUBSTRING((SELECT user()),1,1))>100--",
    "' AND 'a'='a",
    # time based
    "1' AND SLEEP(5)--",
    "' OR SLEEP(5)#",
    "1 AND SLEEP(5)",
    "'; WAITFOR DELAY '0:0:5'--",
    "1'; WAITFOR DELAY '0:0:10'--",
    "IF(1=1,SLEEP(5),0)",
    "' AND (SELECT 1 FROM (SELECT SLEEP(5))a)--",
    "1' AND pg_sleep(5)--",
    "'; SELECT pg_sleep(5);--",
    # error based
    "' AND extractvalue(1,concat(0x7e,(select database())))--",
    "' AND updatexml(1,concat(0x7e,version()),1)--",
    "1' AND GTID_SUBSET(CONCAT(0x7e,database()),0x7e)--",
    "' AND (SELECT * FROM (SELECT RAND()*500000) x)--",
    "1 AND (SELECT 1 FROM(SELECT COUNT(*),CONCAT(version(),FLOOR(RAND(0)*2))x FROM information_schema.tables GROUP BY x)a)--",
    # stacked queries
    "'; DROP TABLE users;--",
    "'; DELETE FROM accounts;--",
    "'; INSERT INTO users(id,user) VALUES(1,'hacker');--",
    "'; EXEC xp_cmdshell('whoami');--",
    "1'; SHOW TABLES;--",
    # order by / column count
    "1 ORDER BY 10--",
    "-1 ORDER BY 100--",
    "1 GROUP BY 5 HAVING 1=1--",
    # others
    "' AND 1=1 UNION SELECT user FROM mysql.user--",
    "1'||(SELECT version())||'",
    "' + (SELECT 'x') + '",
    "admin' AND 1=0 UNION SELECT 'admin','5f4dcc3b5aa765d61d8327deb882cf99'--",
    "%31%27%20OR%20%27%31%27%3D%27%31",
    "1' or '1'='1' or '1'='2",
    "select version()",
    "';SELECT sleep(5);--",
    "1 and extractvalue(rand(),concat(0x3a,database()))",
]

BENIGN = [
    # normal CRUD sql
    "SELECT id, name FROM users WHERE id = 1",
    "select * from products where price < 100 order by name",
    "SELECT username, email FROM accounts WHERE active = 1",
    "INSERT INTO logs (level, message) VALUES ('INFO', 'started')",
    "UPDATE inventory SET quantity = quantity - 1 WHERE sku = 'ABC123'",
    "DELETE FROM sessions WHERE expires_at < NOW()",
    "SELECT COUNT(*) FROM orders WHERE customer_id = 42",
    "SELECT p.name, c.title FROM posts p JOIN categories c ON p.cat_id = c.id",
    "SELECT * FROM employees WHERE department = 'engineering' LIMIT 10",
    "SHOW TABLES",
    "DESCRIBE users",
    "SELECT a.name, b.score FROM teams a LEFT JOIN scores b ON a.id = b.team_id WHERE b.season = 2024",
    # parameterized / placeholder forms
    "SELECT * FROM users WHERE id = ?",
    "SELECT * FROM users WHERE id = %s",
    "INSERT INTO accounts (user_id, role) VALUES (%s, %s)",
    "SELECT username FROM accounts WHERE email = $1 AND active = true",
    "UPDATE t SET col = :value WHERE key = :pk",
    "SELECT * FROM table WHERE name LIKE ? ORDER BY ? DESC",
    "SELECT * FROM foo WHERE x BETWEEN ? AND ?",
    # plain web / api input
    "q=python tutorial",
    "search?keyword=wireless+mouse",
    "?page=2&sort=asc",
    "/products?id=123",
    "https://example.com/api/users?limit=50&offset=0",
    "user: admin logged in at 09:15",
    "order number 100245 shipped",
    "apple banana cherry",
    "Hello world",
    "This is a normal test sentence.",
    "key=value",
    "filter[category]=books&filter[price]=10",
    "name=John&city=New York",
    "client_id=abc123&grant_type=authorization_code",
    "password must contain at least 8 characters",
    "referer=https://example.com/page?from=home",
    "amount=99.99 currency=USD",
    "status=pending",
    "The quick brown fox jumps over the lazy dog",
    "temperature=23.5",
    # tricky-but-benign lookalikes
    "order by 2 limit 1",
    "select name from people where age between 18 and 60 group by name",
    "page 1 of 100 results",
    "we use select * in read-only replicas only",
    "note: the keyword 'admin' triggers no action",
    "or you may choose another option",
    "and then we continue",
    "SELECT 1",
    "1+1=2 and 2+2=4",
]


def main() -> None:
    rows = [("text", "label")]
    rows.extend((t, "1") for t in INJECTIONS)
    rows.extend((t, "0") for t in BENIGN)

    out = Path(__file__).resolve().parent / "synthetic_dataset.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    n_inj = len(INJECTIONS)
    n_ben = len(BENIGN)
    print(f"Wrote {out}")
    print(f"  injections (label=1): {n_inj}")
    print(f"  benign     (label=0): {n_ben}")
    print(f"  total                : {n_inj + n_ben}")


if __name__ == "__main__":
    main()
