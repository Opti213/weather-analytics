SELECT avg(transponed_arrays.element :: numeric)
FROM
    :table_name,
    LATERAL (
        SELECT val ->> length_series.idx element,
               length_series.idx         idx
        FROM (
                 SELECT generate_series(0, jsonb_array_length(val) - 1)
             ) length_series(idx)
        ) transponed_arrays
WHERE ((to_char(dat, 'YY') = to_char(:target::date, 'YY')) and
       (to_char(dat, 'MM') = to_char(:target::date, 'MM')))
GROUP BY transponed_arrays.idx
ORDER BY transponed_arrays.idx;



SELECT avg(transponed_arrays.element :: numeric)
FROM
    :table_name,
    LATERAL (
        SELECT val ->> length_series.idx element,
               length_series.idx         idx
        FROM (
                 SELECT generate_series(0, jsonb_array_length(val) - 1)
             ) length_series(idx)
        ) transponed_arrays
WHERE :table_name.dat BETWEEN :start_date
          AND :end_date
GROUP BY transponed_arrays.idx
ORDER BY transponed_arrays.idx;
