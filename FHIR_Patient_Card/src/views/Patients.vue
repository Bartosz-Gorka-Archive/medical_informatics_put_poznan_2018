<template>
  <main class="l-app__main" role="main">

    <header class="c-toolbar">
      <div class="c-cell">
        <div class="c-cell__media">
          <i class="icon-footprint"></i>
        </div>
        <div class="c-cell__content">
          <h1 class="c-toolbar__title">Patients list</h1>
          <ol class="c-breadcrumb">
            <li class="c-breadcrumb__item">Patients</li>
          </ol>
        </div>
      </div>
    </header>

    <div>
      <div class="l-row u-pb-15">
        <div class="l-col-6@md">
          <input
            v-model="this.familyName"
            @keyup.enter=setFamilyNameFilter(this.familyName)
            type="text"
            class="form-input"
            placeholder="Find patient by family name">
        </div>
      </div>

      <table class="table table--data">
        <thead>
          <tr>
            <th class="u-hiddenDown@md">No</th>
            <th>ID</th>
            <th>Family name</th>
            <th>Gender</th>
            <th>Birthdate</th>
            <th>Active</th>
          </tr>
        </thead>
        <tfoot>
          <tr>
            <th class="u-hiddenDown@md">No</th>
            <th>ID</th>
            <th>Family name</th>
            <th>Gender</th>
            <th>Birthdate</th>
            <th>Active</th>
          </tr>
        </tfoot>
        <tbody>
          <template v-for="(patient, index) in this.patients">
            <tr>
              <td data-label="No" class="u-hiddenDown@md">{{ index + 1 }}</td>
              <td data-label="ID">{{ get(['resource', 'id'], patient) }}</td>
              <td data-label="Family name">{{ get(['resource', 'name', 0, 'family'], patient) }}</td>
              <td data-label="Gender">{{ get(['resource', 'gender'], patient) }}</td>
              <td data-label="Birthdate">{{ get(['resource', 'birthDate'], patient) }}</td>
              <td data-label="Active">{{ get(['resource', 'active'], patient) }}</td>
            </tr>
          </template>

        <infinite-loading
          v-if="loadingPatients"
          v-on:infinite="infiniteHandler"
          ref="infiniteLoading"
          spinner="bubbles">
        </infinite-loading>

        </tbody>
      </table>
    </div>

  </main>
</template>

<script>
  import { createNamespacedHelpers } from 'vuex'
  const { mapGetters, mapActions } = createNamespacedHelpers('patient')

  export default {
    name: 'PatientsView',
    computed: mapGetters(['loadingPatients', 'patients', 'familyName']),
    mounted () {
      mapActions(['clear'])
    },
    methods: {
      get (p, o) {
        return p.reduce((xs, x) => (xs && xs[x]) ? xs[x] : null, o)
      },
      setFamilyNameFilter (name) {
        this.$store.dispatch('patient/setFindByFamilyName', name)
      },
      infiniteHandler (state) {
        this.$store.dispatch('patient/getPatients')
        .then(data => {
          mapGetters(['loadingPatients', 'patients'])
          if (this.loadingPatients) {
            state.loaded()
          } else {
            state.complete()
          }
        })
      }
    }
  }
</script>
