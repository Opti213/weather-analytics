import ast
import asyncio
import io
import random
import struct
from datetime import date, datetime
from typing import Any, List

import matplotlib.pyplot as plt
import pool as pool

from sklearn import svm, datasets, metrics
from sklearn.metrics import roc_curve, auc

import buildpg
import numpy
import numpy as np
import uvicorn
from asyncpg import Connection, connect, pool, create_pool
from fastapi import FastAPI, Form, File, UploadFile
from pydantic import BaseModel
from pypika import Query, Table, Order
from starlette.config import Config
from starlette.requests import Request
from starlette.responses import JSONResponse, HTMLResponse, FileResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

config = Config('env')
DATABASE_URL = config('DATABASE_URL')

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')
templates = Jinja2Templates(directory='templates')


async def xyt_to_feature(x, y, temperature):
    return {
        'lat': y,
        'lng': x,
        'count': float(temperature)
    }


async def get_bigdict_from_matrix(matrix: list):
    xyt = []
    step = 2.5
    c = 0
    for i, y in enumerate(np.arange(90, -90, -step)):
        for j, x in enumerate(
                np.append(np.arange(0, 180, step), np.arange(-180, 0, step))):
            t = matrix[i + j + c]
            val = [x, y, t]
            xyt.append(val)
        c += 143

    bigdict = [await xyt_to_feature(x=val[0], y=val[1], temperature=val[2])
               for val in xyt]
    return bigdict


def get_false_roc_dots():
    y = np.array([random.randint(0, 10) for _ in range(100)])
    scores = np.array([random.randint(2, 7) for _ in range(100)])
    fpr, tpr, _ = metrics.roc_curve(y, scores, pos_label=2)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def get_roc_dots(target, scores):
    fpr, tpr, _ = metrics.roc_curve(target, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def render_roc(fpr, tpr, roc_auc) -> str:
    roc_name = "roc.png"
    plt.figure()
    lw = 2

    plt.plot(
        fpr,
        tpr,
        color='darkorange',
        lw=lw,
        label='ROC curve (area = %0.2f)' % roc_auc
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(roc_name)
    return roc_name


def read_target(target_bytes: bytes) -> numpy.array:
    def target_gen(target_bytes: bytes = target_bytes):
        data = io.ByteIO(target_bytes)
        while True:
            yield struct.unpack('f', data.read(4))[0]

    target = target_gen()

    # xdef 144 linear 0.000000 2.500000
    # ydef 73 linear -90.000000 2.500000
    # zdef 29 linear 1 1
    # tdef 6 linear 1feb2012  1dy
    # vars 5
    # mslp    29 99 [hPa] Mean Sea Level Pressure - CLIM + ANOM
    # h500    29 99 [m]
    # tsrf    29 99 [K]
    # t850    29 99 [K]
    # Prec    29 99 Precip []
    # params = [vars][days][z][y][x]
    params = ["prec", "t850", "tsrf", "h500", "mslp"]
    res = dict()
    for var in params:
        param = []
        for day in range(6):
            one_day = []
            for z in range(29):
                z_row = []
                for y in range(73):
                    y_row = [next(target) for x in range(144)]
                    z_row.append(y_row)
                one_day.append(z_row)
            param.append(one_day)
        res[var] = numpy.array(param)
    return res


@app.get('/')
async def index(request: Request) -> templates.TemplateResponse:
    table_names = ['h500', 'merd', 'prec', 't850']
    context = {
        'request': request,
        'table_names': table_names
    }
    return templates.TemplateResponse('index.html', context)


@app.get('/roc')
async def upload_roc(request: Request) -> HTMLResponse:
    context = {
        'request': request,
    }
    return templates.TemplateResponse("roc.html", context)


@app.post('/roc')
async def upload_roc(*, request: Request, f: bytes = File(...),
                     type_of_data: str = Form(...)) -> Any:
    # todo correct data
    # target = read_target(f)
    # fpr, tpr, roc_auc = get_roc_dots()
    fpr, tpr, roc_auc = get_false_roc_dots()
    roc = render_roc(fpr, tpr, roc_auc)
    context = {
        "request": request,
        "links": ["roc_l.png", "roc_m.png", "roc_h.png"]}
    return templates.TemplateResponse('links.html', context=context)


@app.get('/tables/{table_name}')
async def table_view(request: Request, table_name: str) -> templates.TemplateResponse:
    context = {
        'request': request,
        'table_name': table_name,
    }
    return templates.TemplateResponse('table.html', context)


@app.get('/tables/{table_name}/records/{record_id}')
async def record_view(
        request: Request,
        table_name: str,
        record_id: int
) -> templates.TemplateResponse:
    conn: Connection = app.state.connection
    table = Table(table_name)
    record = await conn.fetchrow(
        str((Query.from_(table).select(table.dat).where(table.id == record_id)))
    )
    context = {
        'request': request,
        'table_name': table_name,
        'record_id': record_id,
        'dat': record[0],
    }
    return templates.TemplateResponse('record.html', context)


@app.get('/tables/{table_name}/records/{record_id}/bigdict/')
async def get_bigdict(table_name: str, record_id: int):
    conn: Connection = app.state.connection
    table = Table(table_name)
    record = await conn.fetchrow(
        str((Query.from_(table)
             .select(table.val)
             .where(table.id == record_id)))
    )
    matrix = record[0]
    matrix = matrix[1:-1]
    matrix = list(matrix.split(', '))
    bigdict = await get_bigdict_from_matrix(matrix)
    return JSONResponse(bigdict)


@app.post('/tables/{table_name}/get_average_for_values/')
async def get_average_for_values(
        *,
        table_name: str,
        start_date: date = Form(...),
        end_date: date = Form(...),
) -> JSONResponse:
    conn = app.state.connection
    query, args = buildpg.render(
        """
        SELECT
            avg(transponed_arrays.element :: numeric)
        FROM
            :table_name,
            LATERAL (
                SELECT
                    val ->> length_series.idx element,
                    length_series.idx idx
                FROM
                    (
                        SELECT
                            generate_series(0, jsonb_array_length(val) - 1)
                    ) length_series(idx)
            ) transponed_arrays
        WHERE
            :table_name.dat BETWEEN :start_date
            AND :end_date
        GROUP BY
            transponed_arrays.idx
        ORDER BY
            transponed_arrays.idx;
        """,
        table_name=buildpg.V(table_name),
        start_date=start_date,
        end_date=end_date,
    )
    calculated_values = await conn.fetch(query, *args)
    values = [rec[0] for rec in calculated_values]
    bigdict = await get_bigdict_from_matrix(values)
    return JSONResponse(bigdict)


@app.get('/tables/{table_name}/{record_id}/anomaly/by_date')
async def get_anomaly(
        *,
        table_name: str,
        start_date: date = Form(...),
        end_date: date = Form(...),
        request: Request):
    pool = app.state.pool

    query, args = buildpg.render(
        """
        SELECT
            avg(transponed_arrays.element :: numeric)
        FROM
            :table_name,
            LATERAL (
                SELECT
                    val ->> length_series.idx element,
                    length_series.idx idx
                FROM
                    (
                        SELECT
                            generate_series(0, jsonb_array_length(val) - 1)
                    ) length_series(idx)
            ) transponed_arrays
        WHERE
            :table_name.dat BETWEEN :start_date
            AND :end_date
        GROUP BY
            transponed_arrays.idx
        ORDER BY
            transponed_arrays.idx;
        """,
        table_name=buildpg.V(table_name),
        start_date=start_date,
        end_date=end_date,
    )
    async with pool.acquire() as conn:
        calculated_values = await conn.fetch(query, *args)
        current_day = [rec[0] for rec in calculated_values]

    # todo average for all time and calculate anomaly
    async with pool.acquire() as conn:
        calculated_values = await conn.fetch(query, *args)
        avg_values = [rec[0] for rec in calculated_values]
    anomaly = [int(a) - int(b) for a, b in zip(current_day, avg_values)]
    bigdict = await get_bigdict_from_matrix(anomaly)
    return JSONResponse(bigdict)


@app.get('/tables/{table_name}/records/{record_id}/anomaly/')
async def get_anomaly(table_name: str, record_id: int):
    pool = app.state.pool
    table = Table(table_name)
    async with pool.acquire() as conn:
        record = await conn.fetchrow(
            str((Query.from_(table)
                 .select(table.val, table.dat)
                 .where(table.id == record_id)))
        )
        target_date = record[1]
        matrix = record[0]
        matrix = matrix[1:-1]
        current_day = list(matrix.split(', '))
    query, args = buildpg.render(
        """
        SELECT
            avg(transponed_arrays.element :: numeric)
        FROM
            :table_name,
            LATERAL (
                SELECT
                    val ->> length_series.idx element,
                    length_series.idx idx
                FROM
                    (
                        SELECT
                            generate_series(0, jsonb_array_length(val) - 1)
                    ) length_series(idx)
            ) transponed_arrays
        WHERE (to_char(dat, 'DD') = to_char(:target::date, 'DD') and to_char(dat, 'MM') = to_char(:target::date, 'MM') )
        GROUP BY
            transponed_arrays.idx
        ORDER BY
            transponed_arrays.idx;
        """,
        table_name=buildpg.V(table_name),
        target=target_date
    )
    async with pool.acquire() as conn:
        calculated_values = await conn.fetch(query, *args)
        avg_values = [rec[0] for rec in calculated_values]
    anomaly = [int(a) - int(b) for a, b in zip(current_day, avg_values)]
    bigdict = await get_bigdict_from_matrix(anomaly)
    return JSONResponse(bigdict)


@app.post('/tables/{table_name}/average/')
async def average(
        *,
        table_name: str,
        start_date: date = Form(...),
        end_date: date = Form(...),
        request: Request) -> templates.TemplateResponse:
    context = {
        'request': request,
        'table_name': table_name,
        'start_date': start_date,
        'end_date': end_date,
    }
    return templates.TemplateResponse('average.html', context)


@app.route('/error')
async def error(request):
    raise RuntimeError('Oh no')


@app.exception_handler(404)
async def not_found(request, exc):
    template = '404.html'
    context = {'request': request}
    return templates.TemplateResponse(template, context, status_code=404)


@app.exception_handler(500)
async def server_error(request, exc):
    template = '500.html'
    context = {'request': request}
    return templates.TemplateResponse(template, context, status_code=500)


@app.on_event('startup')
async def app_init():
    app.state.pool = await create_pool(DATABASE_URL)
    app.state.connection = await connect(DATABASE_URL)


@app.on_event('shutdown')
async def app_stop():
    await app.state.pool.close()
    await app.state.connection.close()


if __name__ == '__main__':
    uvicorn.run('app:app', host='localhost', port=8000, reload=False)
