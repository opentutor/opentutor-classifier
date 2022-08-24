#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#

import json
import os
import shutil
import responses

from tests import fixture_path
from tests.utils import mocked_data_dao


@responses.activate
def test_delete_q1(client):
    mocked_model_path = os.path.join(".", "mockModels", "fixtures", "models")
    shutil.copytree(fixture_path("models"), mocked_model_path)

    with mocked_data_dao(
        "q1",
        fixture_path("data"),
        os.path.abspath(mocked_model_path),
        fixture_path("models_deployed"),
    ):
        res = client.post(
            "/classifier/delete_model/",
            data=json.dumps({"lesson": "q1"}),
            content_type="application/json",
        )
        assert res.status_code == 200

    shutil.rmtree(os.path.join(".", "mockModels"))
