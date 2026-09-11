#!/usr/bin/env julia

include(joinpath(@__DIR__, "fresh_process_campaign.jl"))
using .FreshProcessCampaign

FreshProcessCampaign.campaign_main(ARGS)
