#
# This software is Copyright ©️ 2020 The University of Southern California. All Rights Reserved.
# Permission to use, copy, modify, and distribute this software and its documentation for educational, research and non-profit purposes, without fee, and without a written agreement is hereby granted, provided that the above copyright notice and subject to the full license file found in the root of this software deliverable. Permission to make commercial use of this software may be obtained by contacting:  USC Stevens Center for Innovation University of Southern California 1150 S. Olive Street, Suite 2300, Los Angeles, CA 90115, USA Email: accounting@stevens.usc.edu
#
# The full terms of this copyright and license should always be found in the root directory of this software deliverable as "license.txt" and if these terms are not found with this software, please contact the USC Stevens Center for the full license.
#
import responses

import opentutor_classifier
from opentutor_classifier import (
    FeaturesSaveRequest,
    ExpectationFeatures,
)
from opentutor_classifier.api import get_graphql_endpoint, update_features_gql


@responses.activate
def test_saves_features_to_gql():
    req = FeaturesSaveRequest(
        lesson="lesson1",
        expectations=[
            ExpectationFeatures(expectation=2, features=dict(good=["a", "b"], bad="c"))
        ],
    )
    responses.add(
        responses.POST,
        get_graphql_endpoint(),
        json=update_features_gql(req),
        status=200,
    )
    opentutor_classifier.find_data_dao().save_features(req)
