DROP TABLE IF EXISTS h500;
DROP TABLE IF EXISTS merd;
DROP TABLE IF EXISTS prec;
DROP TABLE IF EXISTS t850;

CREATE TABLE h500 (
    id serial,
    val JSONB, dat timestamp,
    primary key (id));

    INSERT INTO
        h500 (val, dat)
    SELECT
        array_to_json(
            array(
                SELECT
                    floor(random() * (300 -200 + 1) + 200) :: int
                FROM
                    generate_series(0, 144 * 73 -1)
            )
        ),
        dd.dat
    FROM
        generate_series (
            '2019-01-01' :: timestamp,
            '2020-01-01' :: timestamp,
            '1 day' :: INTERVAL
        ) dd(dat);

CREATE TABLE merd AS (SELECT * FROM h500);
CREATE TABLE prec AS (SELECT * FROM h500);
CREATE TABLE t850 AS (SELECT * FROM h500);